def train(self, mode=True):
    # existing code...
    output = super().forward(*args, **kwargs)
    if torch.isnan(output.logits).any():
        print("Warning: NaN detected in model output logits during training")
    return output
    // ...

def eval(self):
    # existing code...
    output = super().forward(*args, **kwargs)
    if torch.isnan(output.logits).any():
        print("Warning: NaN detected in model output logits during evaluation")
    return output
    // ... 