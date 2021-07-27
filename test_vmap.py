from torch import vmap
batch_size, feature_size = 3, 5

weights = torch.randn(feature_size, requires_grad=True)

def model(feature_vec):
  assert feature_vec.dim() == 1
  return feature_vec.dot(weights).relu()

def compute_loss(example, target):
  y = model(example)
  return ((y - target) ** 2).mean()  # MSELoss

examples = torch.randn(batch_size, feature_size)
targets = torch.randn(batch_size)

results = vmap(compute_loss)(examples, target)

