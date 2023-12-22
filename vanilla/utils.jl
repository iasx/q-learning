"""
  Shared utility functions.
"""

# *----------------------------------------------------------------------------* Exponential Decay

"Exponential decay function."
decay(
  x::Int,
  λ::Float64=1e-1;
  max::Float64=1.0,
  min::Float64=0.05,
)::Float64 = (max - min)exp(-λ * x) + min

# *----------------------------------------------------------------------------* Pretty Printing

"Print a line separator."
printline() = println("-"^120)

"Print a line."
printspace() = println("\n"^2)
