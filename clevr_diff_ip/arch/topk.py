import torch

# class Square(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         # Because we are saving one of the inputs use `save_for_backward`
#         # Save non-tensors and non-inputs/non-outputs directly on ctx
#         ctx.save_for_backward(x)
#         return x**2
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         # A function support double backward automatically if autograd
#         # is able to record the computations performed in backward
#         x, = ctx.saved_tensors
#         return grad_out * 2 * x
#
# # Use double precision because finite differencing method magnifies errors
# x = torch.rand(3, 3, requires_grad=True, dtype=torch.double)
# torch.autograd.gradcheck(Square.apply, x)
# # Use gradcheck to verify second-order derivatives
# torch.autograd.gradgradcheck(Square.apply, x)

# import torchviz
#
# x = torch.tensor(1., requires_grad=True).clone()
# out = Square.apply(x)
# grad_x, = torch.autograd.grad(out, x, create_graph=True)
# torchviz.make_dot((grad_x, x, out), {"grad_x": grad_x, "x": x, "out": out})

# class Exp(torch.autograd.Function):
#     # Simple case where everything goes well
#     @staticmethod
#     def forward(ctx, x):
#         # This time we save the output
#         result = torch.exp(x)
#         # we should use `save_for_backward` here when
#         # the tensor saved is an ouptut (or an input).
#         ctx.save_for_backward(result)
#         return result
#
#     @staticmethod
#     def backward(ctx, grad_out):
#         result, = ctx.saved_tensors
#         return result * grad_out
#
# x = torch.tensor(1., requires_grad=True, dtype=torch.double).clone()
# # Validate our
# gradients using gradcheck
# torch.autograd.gradcheck(Exp.apply, x)
# torch.autograd.gradgradcheck(Exp.apply, x)

## Jax perturbed TopK
# def sorted_topk_indicators(x, k, sort_by = SortBy.POSITION):
#   """Finds the (sorted) positions of the topk values in x.
#   Args:
#     x: The input scores of dimension (d,).
#     k: The number of top elements to find.
#     sort_by: Strategy to order the extracted values. This is useful when this
#       function is applied to many perturbed input and average. As topk's output
#       does not have a fixed order, the indicator vectors could be swaped and the
#       average of the indicators would not be spiky.
#   Returns:
#     Indicator vectors in a tensor of shape (k, d)
#   """
#   n = x.shape[-1]
#   values, ranks = jax.lax.top_k(x, k)
#
#   if sort_by == SortBy.NONE:
#     sorted_ranks = ranks
#   if sort_by == SortBy.VALUES:
#     sorted_ranks = ranks[jnp.argsort(values)]
#   if sort_by == SortBy.POSITION:
#     sorted_ranks = jnp.sort(ranks)
#
#   one_hot_fn = jax.vmap(functools.partial(jax.nn.one_hot, num_classes=n))
#   indicators = one_hot_fn(sorted_ranks)
#   return indicators
#
#
# def perturbed(func,
#               num_samples = 1000,
#               noise = Noise.NORMAL):
#   """Creates a function that applies func on multiple perturbed input.
#   Args:
#     func: The function to make a perturbed version of.
#     num_samples: The number of perturbed input to generate, pass through func
#       and average to obtain the perturbed output.
#     noise: Type of the noise.
#   Returns:
#     A function with the same signature as `func` but that will compute an
#     expectation of the perturbed output.
#   """
#   noise = Noise(noise)
#   assert noise == Noise.NORMAL, "Only normal noise is supported for now."
#
#   @jax.custom_vjp
#   def foo(input_tensor, sigma, rng_key):
#     return forward(input_tensor, sigma, rng_key)[0]
#
#   def forward(input_tensor, sigma, rng_key):
#     noise_shape = (num_samples,) + input_tensor.shape
#     noise = jax.random.normal(rng_key, shape=noise_shape)
#     noise_gradient = noise
#     noisy_input_tensor = input_tensor + noise * sigma
#     perturbed_outputs = jax.vmap(func)(noisy_input_tensor)
#     forward_outputs = perturbed_outputs.mean(axis=0)
#     keep_for_bwd = (perturbed_outputs, noise_gradient, sigma)
#     return forward_outputs, keep_for_bwd
#
#   def backward(keep_for_bwd, output_grad):
#     perturbed_outputs, noise_gradient, sigma = keep_for_bwd
#     expected_gradient = jnp.mean(
#         perturbed_outputs * noise_gradient[:, None, :] / sigma, axis=0)
#     return ((output_grad * expected_gradient).sum(axis=0), None, None)
#
#   foo.defvjp(forward, backward)
#   return foo
#
#
# def perturbed_sorted_topk_indicators(x, rng, k,
#                                      sigma,
#                                      num_samples = 1000,
#                                      noise = "normal"):
#   return perturbed(
#       functools.partial(sorted_topk_indicators, k=k, sort_by=SortBy.POSITION),
#       num_samples, noise)(x, sigma, rng)