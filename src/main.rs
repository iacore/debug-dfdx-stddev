use std::{fs::File, io::Read};

use dfdx::prelude::*;
fn main() {
    let mut buffer = [0u8; 1024 * 4];
    let mut f = File::open("emb[688].raw").unwrap();
    f.read_exact(&mut buffer).unwrap();

    let dev = Cpu::default();
    let mut a: Tensor1D<1024> = dev.zeros();
    a.copy_from(unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const f32, 1024) });
    let shape = a.shape().concrete();

    let x = (a.clone() - a.mean().broadcast()).square();
    let std = sqrt(x.mean());

    println!("{:?} {:.17?}", shape, std.array());
}
