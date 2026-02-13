class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
