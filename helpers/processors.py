from transformers import LogitsProcessor, StoppingCriteria, LogitsProcessorList, StoppingCriteriaList
import torch

# Custom Logits Processor to boost specific token logits
class BoostLogitsProcessor(LogitsProcessor):
    def __init__(self, boost_token_ids, boost_value):
        self.boost_token_ids = boost_token_ids
        self.boost_value = boost_value

    def __call__(self, input_ids: torch.LongTensor, logits: torch.FloatTensor) -> torch.FloatTensor:
        logits[:, self.boost_token_ids] += self.boost_value
        return logits
    
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_sequences):
        super().__init__()
        self.stop_sequences = stop_sequences  # List of token ID sequences

    def __call__(self, input_ids, scores, **kwargs):
        for stop_seq in self.stop_sequences:
            if len(input_ids[0]) >= len(stop_seq):
                if torch.equal(input_ids[0][-len(stop_seq):], torch.tensor(stop_seq).to(input_ids.device)):
                    return True
        return False

def create_boost_processor(tokenizer, boost_words, boost_value):
    if boost_words:
        boost_token_ids = []
        for word in boost_words:
            tokens = tokenizer(word).input_ids
            if tokenizer.unk_token_id in tokens:
                print(f"Warning: Word '{word}' contains unknown tokens.")
            boost_token_ids.extend(tokens)

        boost_token_ids = list(set(boost_token_ids))
        boost_processor = BoostLogitsProcessor(boost_token_ids, boost_value)
        return LogitsProcessorList([boost_processor])
    else:
        return LogitsProcessorList()  # Return an empty processor if no boost words

# class StopAfterNSentences(StoppingCriteria):
#     def __init__(self, tokenizer, n_sentences=2):
#         super().__init__()
#         self.tokenizer = tokenizer
#         self.n_sentences = n_sentences
#         self.sentence_count = 0

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
#         text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
#         # Count number of sentence enders
#         self.sentence_count = text.count('.') + text.count('!') + text.count('?')
#         return self.sentence_count >= self.n_sentences

# def create_stopping_criteria(tokenizer):
#     return StoppingCriteriaList([
#         StopAfterNSentences(tokenizer, n_sentences=2)
#     ])

def create_stopping_criteria(tokenizer):
    stop_sequences = [
        tokenizer.encode("User:", add_special_tokens=False),
        tokenizer.encode("\nUser:", add_special_tokens=False),
        tokenizer.encode("Assistant:", add_special_tokens=False),
        tokenizer.encode("\nAssistant:", add_special_tokens=False)
    ]
    return StoppingCriteriaList([StopOnTokens(stop_sequences)])
