using Reel
using Images

const Pixel = RGB{N0f8}
const Scene = Matrix{Pixel}

# *----------------------------------------------------------------------------*

function tile2color(T::Tile)::Pixel
  if T == Goal
    return colorant"green"
  elseif T == Hole
    return colorant"red"
  elseif T == None
    return colorant"white"
  end
end


world2image(w::World)::Scene = colorview(RGB, tile2color.(w))


function draw(world::World, steps::Steps)::Vector{Scene}
  scenes::Vector{Scene} = [world2image(world) for _ in 1:length(steps)]

  for (i, (x, y)) in enumerate(steps)
    scenes[i][x, y] = colorant"blue"
  end

  return scenes
end


function save(scenes::Vector{Scene}; framerate::Int=10, name::String="result")
  frames = Frames(MIME("image/png"), fps=framerate)

  for s in scenes
    push!(frames, s)
  end

  write("$name.gif", frames)
end


save(
  world::World,
  steps::Steps;
  framerate::Int=10,
  name::String="result",
) = save(draw(world, steps); name, framerate)
