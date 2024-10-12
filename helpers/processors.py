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

def create_stopping_criteria(tokenizer):
    stop_sequences = [
        tokenizer.encode("\nUser:", add_special_tokens=False),
        tokenizer.encode("\nAssistant:", add_special_tokens=False)
    ]
    return StoppingCriteriaList([StopOnTokens(stop_sequences)])
