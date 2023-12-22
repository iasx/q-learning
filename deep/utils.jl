"""
  Shared utility functions.
"""

# *----------------------------------------------------------------------------* Exponential Decay

"Exponential decay function."
decay(
  x::Int,
  λ::Float32=1f-1;
  max::Float32=1f0,
  min::Float32=5f-2,
)::Float32 = (max - min)exp(-λ * x) + min

# *----------------------------------------------------------------------------* Pretty Printing

"Print a line separator."
printline() = println("-"^120)

"Print a line."
printspace() = println("\n"^2)