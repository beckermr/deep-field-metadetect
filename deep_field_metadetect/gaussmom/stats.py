import jax
import jax.numpy as jnp


@jax.jit
def get_ratio_var(a, b, var_a, var_b, cov_ab):
    """
    Compute the variance of (a/b).
    """

    # Ensure no division by zero
    b = jnp.where(
        b == 0, 1e-100, b
    )  # TODO: This does not raise a value error like ngmix

    rsq = (a / b) ** 2

    var = rsq * (var_a / a**2 + var_b / b**2 - 2 * cov_ab / (a * b))

    var = jnp.where(var > 1e20, jnp.nan, var)
    return var
