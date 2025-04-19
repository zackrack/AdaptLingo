import os
import time
import json
import re
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss

import nltk
import soundfile as sf
from sentence_transformers import SentenceTransformer

# ——— Logging & Device Setup ———
logging.getLogger("transformers").setLevel(logging.ERROR)
MODEL_SAVE_DIR = "/local/speech/users/wl2904"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ——— TTS Helpers ———
def ensure_nltk_tagger_resource():
    """Make sure NLTK has the averaged_perceptron_tagger."""
    for tag in [
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger_eng",
    ]:
        try:
            nltk.data.find(tag)
        except LookupError:
            print(f"Downloading NLTK resource {tag}…")
            nltk.download(tag.split("/")[-1])
            print("Download complete.")


def generate_speech_from_text(model, text, filename=None, language="EN"):
    ensure_nltk_tagger_resource()
    os.makedirs("static/audio", exist_ok=True)
    try:
        sr, audio = model.infer(text=text, language=language)
        if not filename:
            filename = f"assistant_response_{int(time.time())}.wav"
        path = os.path.join("static", "audio", filename)
        sf.write(path, audio, sr)
        print(f"Saved TTS to {path}")
        return path
    except LookupError as e:
        print("Missing resource, retrying…", e)
        ensure_nltk_tagger_resource()
        return generate_speech_from_text(model, text, filename, language)
    except Exception as e:
        print("TTS error:", e)
        return None


# ——— Unified Generation API ———
def generate_response(
    model,
    tokenizer,
    prompt: str,
    logits_processor=None,
    stopping_criteria=None,
    device=None,
) -> str:
    """
    Dispatch to RAG if available, otherwise run legacy HF generate.
    """
    # RAG path
    if hasattr(model, "generate_rag_response"):
        return model.generate_rag_response(
            prompt,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            device=device,
        )

    # legacy HF-only path
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    reply = text[len(prompt) :].split("\nUser:")[0].strip()
    return re.split(r"(User:|Assistant:)", reply)[0].strip()


# ——— RAG Conversation System ———
class RAGConversationSystem:
    def __init__(
        self,
        embedding_model_name="all-MiniLM-L6-v2",
        tokenizer=None,
        llm_model=None,
        max_history=20,
    ):
        """
        Args:
          embedding_model_name: name of the sentence-transformers embedder
          tokenizer:           a HuggingFace AutoTokenizer already initialized
          llm_model:           a HuggingFace AutoModelForCausalLM already on device
          max_history:         how many past turns to keep
        """
        # Sentence embeddings
        self.embedding_model = SentenceTransformer(
            embedding_model_name, cache_folder=MODEL_SAVE_DIR
        )
        self.embedding_size = self.embedding_model.get_sentence_embedding_dimension()

        # LLM & tokenizer (must be provided)
        if tokenizer is None or llm_model is None:
            raise ValueError(
                "You must pass both tokenizer and llm_model to RAGConversationSystem"
            )
        self.tokenizer = tokenizer
        self.llm_model = llm_model.to(device)

        # History / FAISS index
        self.conversation_history = []
        self.embeddings = np.empty((0, self.embedding_size))
        self.faiss_index = faiss.IndexFlatL2(self.embedding_size)
        if device.type == "cuda":
            res = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        # Attention fusion
        self.attention = nn.Linear(self.embedding_size, 1).to(device)
        self.max_history = max_history

    def _compute_attention_weights(self, query_emb, ctx_embs):
        combined = query_emb.unsqueeze(0) * ctx_embs  # (N, D)
        scores = self.attention(combined).squeeze(-1)  # (N,)
        return F.softmax(scores, dim=0)

    def _retrieve_static_docs(self, query, top_k=2):
        return [
            {"content": f"Static doc about '{query[:20]}...'", "source": "Wikipedia"}
        ]

    def retrieve_relevant_context(self, query, top_k=3):
        q_emb = self.embedding_model.encode(query, convert_to_tensor=True).to(device)
        static = self._retrieve_static_docs(query)
        dynamic = []
        if self.embeddings.shape[0] > 0:
            _, idxs = self.faiss_index.search(q_emb.cpu().numpy()[None, :], top_k)
            for i in idxs[0]:
                if i < len(self.conversation_history):
                    dynamic.append(
                        {"type": "history", "content": self.conversation_history[i]}
                    )
        all_ctx = [{"type": "static", "content": d} for d in static] + dynamic
        if not all_ctx:
            return []
        ctx_embs = torch.stack(
            [
                self.embedding_model.encode(str(c["content"]), convert_to_tensor=True)
                for c in all_ctx
            ]
        ).to(device)
        weights = self._compute_attention_weights(q_emb, ctx_embs)
        paired = sorted(
            zip(all_ctx, weights.cpu().detach().numpy()),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return paired

    def _build_history_block(self):
        lines = []
        for turn in self.conversation_history[-self.max_history :]:
            lines.append(f"User: {turn['user']}\nModel: {turn['model']}")
        return "\n\n".join(lines)

    def generate_rag_response(
        self, query: str, logits_processor=None, stopping_criteria=None, device=None
    ) -> str:
        """
        Generate using the RAG system with optional logits/stopping hooks.
        """
        # retrieve and build prompt
        wc = self.retrieve_relevant_context(query)
        history = self._build_history_block() or "None"
        ctx_lines = []
        for ctx, w in wc:
            if ctx["type"] == "static":
                c = ctx["content"]
                ctx_lines.append(f"[{w:.2f}] {c['content']} (Source: {c['source']})")
            else:
                h = ctx["content"]
                ctx_lines.append(
                    f"[{w:.2f}] Hist → User: {h['user']} | Model: {h['model']}"
                )
        ctx_text = "\n".join(ctx_lines) or "None"

        prompt = "\n\n".join(
            [
                "You are a context‑aware assistant using both history and knowledge.",
                f"History:\n{history}",
                f"Context:\n{ctx_text}",
                f"User: {query}",
                "Model:",
            ]
        )
        # generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(
            device or self.llm_model.device
        )
        length = inputs.input_ids.shape[1]
        max_new = min(150, self.llm_model.config.max_position_embeddings - length - 1)
        if max_new <= 0:
            raise ValueError("Prompt too long.")

        out = self.llm_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.2,
            pad_token_id=self.tokenizer.eos_token_id,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
        )
        raw = self.tokenizer.decode(out[0][length:], skip_special_tokens=True)
        # clean up any trailing User:/Assistant: artifacts
        reply = raw.split("\nUser:")[0].strip()
        return re.split(r"(User:|Assistant:)", reply)[0].strip()

    def update_history(self, user_q, model_r):
        self.conversation_history.append({"user": user_q, "model": model_r})
        emb = self.embedding_model.encode(f"User: {user_q}\nModel: {model_r}")
        emb = emb.reshape(1, -1)
        if len(self.conversation_history) > self.max_history:
            # rebuild
            self.conversation_history.pop(0)
            self.embeddings = np.vstack(
                [
                    self.embedding_model.encode(
                        f"User: {t['user']}\nModel: {t['model']}"
                    )
                    for t in self.conversation_history
                ]
            )
            self.faiss_index.reset()
            if self.embeddings.size:
                self.faiss_index.add(self.embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
            self.faiss_index.add(emb)

    def save_history(self, path="conversation_history.json"):
        with open(path, "w") as f:
            json.dump(self.conversation_history, f)

    def load_history(self, path="conversation_history.json"):
        try:
            with open(path) as f:
                self.conversation_history = json.load(f)
            embs = [
                self.embedding_model.encode(f"User: {t['user']}\nModel: {t['model']}")
                for t in self.conversation_history[-self.max_history :]
            ]
            if embs:
                self.embeddings = np.vstack(embs)
                self.faiss_index.reset()
                self.faiss_index.add(self.embeddings)
        except FileNotFoundError:
            pass
