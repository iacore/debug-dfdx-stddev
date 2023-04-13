import arraymancer
import std/memfiles

var a: Tensor[float32] = newTensor[float32](1024)

var mm = memfiles.open("emb[688].raw", mode = fmRead)

let mf32 = cast[ptr UncheckedArray[float32]](mm.mem)


for i in 0..<1024:
  a[i] = mf32[i]


let x = (a - a.mean.broadcast(1024)).square

# .pow(2.toTensor.broadcast)
    # let x = (a.clone() - a.clone().mean().broadcast()).square();
# let std = sqrt(x.mean());
let std = x.mean.sqrt

echo (std)
