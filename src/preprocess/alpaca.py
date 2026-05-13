from datasets import Features, Value

alpaca_features = Features(
    instruction=Value("string"),
    input=Value("string"),
    output=Value("string"),
    _qid=Value("string"),
)
