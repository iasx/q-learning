"""
  Structures and constants.
"""

# *----------------------------------------------------------------------------* Environment

"Tiles of the world."
@enum Tile begin
  None = 0
  Goal = 5
  Wall = -1
  Hole = -5
end

Base.float(C::Tile)::Float32 = float(Int(C))

# "World perception from agent's point of view."
# struct State
#   N::Tile # top
#   S::Tile # bottom
#   W::Tile # left
#   E::Tile # right
# end

"Map of the world."
const World = Matrix{Tile}

# *----------------------------------------------------------------------------* Actions

"Possible moves."
@enum Move begin
  Up = 1
  Down = 2
  Left = 3
  Right = 4
end

Base.to_index(M::Move)::Int = Int(M)

"List of all possible actions."
const Moves::Vector{Move} = [Up, Down, Left, Right]

# *----------------------------------------------------------------------------* Agent

"Agent's memory."
const QTable = Array{Float32,3}

"Agent's path."
const Steps = Vector{Tuple{Int,Int}}

"Acting agent."
mutable struct Agent
  Q::QTable
  X::Int
  Y::Int

  Agent(
    Q::QTable,
    X::Int=1,
    Y::Int=1,
  ) = new(Q, X, Y)

  Agent(
    Nx::Int,
    Ny::Int,
    M::Int,
    X::Int=1,
    Y::Int=1,
  ) = new(zeros(Nx, Ny, M), X, Y)
end

"Action result."
struct Result
  X::Int
  Y::Int
  R::Float32
end
