extern crate ocl;

use std::io::Read;
use std::io::Write;

const DEAD: ocl::prm::cl_char = 0;
const ALIVE: ocl::prm::cl_char = 1;
const FINALSTATEFILE: &str = "final_state.dat";

fn print_board(board: Vec<ocl::prm::cl_char>, nx: ocl::prm::cl_uint) {
    for (idx, field) in board.iter().enumerate() {
        if idx % nx as usize == 0 {
            println!("");
        }
        print!(
            "{}",
            match field {
                &ALIVE => "O",
                _ => ".",
            }
        );
    }
}

fn post_rustbook() -> Result<(), ocl::error::Error> {
    let args: Vec<std::string::String> = std::env::args().collect();

    let mut tmp = String::new();
    std::fs::File::open(&args[2])?.read_to_string(&mut tmp)?;
    let params: Vec<usize> = tmp.lines()
        .map(|value| value.parse::<usize>().unwrap())
        .collect();
    let (nx, ny, iterations) = (
        params[0] as ocl::prm::cl_uint,
        params[1] as ocl::prm::cl_uint,
        params[2],
    );
    let (bx, by) = (
        args[3].parse::<usize>().unwrap(),
        args[4].parse::<usize>().unwrap(),
    );

    tmp = String::new();
    std::fs::File::open(&args[1])?.read_to_string(&mut tmp)?;

    // maybe there's a more idiomatic way to construct this board
    let mut h_board: Vec<ocl::prm::cl_char> = vec![0; (nx * ny) as usize];
    for field in tmp.lines() {
        let field_info: Vec<usize> = field
            .split_whitespace()
            .map(|value| value.parse::<usize>().unwrap())
            .collect();
        h_board[field_info[0] + field_info[1] * nx as usize] = field_info[2] as ocl::prm::cl_char;
    }

    let mut kernel_src = String::new();
    std::fs::File::open("../gameoflife.cl")?.read_to_string(&mut kernel_src)?;
    let proque = ocl::ProQue::builder().src(kernel_src.clone()).build()?;

    let mut d_board_tick = ocl::Buffer::<ocl::prm::cl_char>::builder()
        .queue(proque.queue().clone())
        .len(h_board.len())
        .flags(ocl::flags::MemFlags::new().read_write())
        .build()?;
    d_board_tick.write(&h_board).enq()?;

    let mut d_board_tock = ocl::Buffer::<ocl::prm::cl_char>::builder()
        .queue(proque.queue().clone())
        .len(h_board.len())
        .flags(ocl::flags::MemFlags::new().read_write())
        .build()?;

    let mut kernel = proque
        .create_kernel("accelerate_life")?
        .gws((nx, ny))
        .lws((bx, by))
        .arg_buf_named("tick", Some(&d_board_tick))
        .arg_buf_named("tock", Some(&d_board_tock))
        .arg_scl(nx)
        .arg_scl(ny)
        .arg_loc::<ocl::prm::cl_char>((bx + 2) * (by + 2));

    println!("Starting state");
    print_board(h_board.clone(), nx);

    for _i in 0..iterations {
        unsafe {
            kernel.enq()?;
        }
        let tmp = d_board_tick;
        d_board_tick = d_board_tock;
        d_board_tock = tmp;

        kernel.set_arg_buf_named("tick", Some(&d_board_tick))?;
        kernel.set_arg_buf_named("tock", Some(&d_board_tock))?;
    }

    d_board_tick.read(&mut h_board).enq()?;

    println!("Final state");
    print_board(h_board.clone(), nx);

    let mut writer = std::fs::File::create(FINALSTATEFILE)?;
    h_board
        .iter()
        .enumerate()
        .filter(|&(_idx, field)| field == &ALIVE)
        .map(|(idx, _field)| {
            writeln!(
                &mut writer,
                "{} {} {}",
                idx % nx as usize,
                idx / nx as usize,
                ALIVE
            )
        })
        .count();

    Ok(())
}

fn main() {
    match std::env::args().count() {
        5 => post_rustbook().unwrap(),
        _ => println!(
            "Usage: {} input.dat input.params bx by
\tinput.dat\tpattern file              
\tinput.params\t parameter file defining board size
\tbx by\t sizes of the thread blocks - must divide the board size equally
        ",
            std::env::args().nth(0).unwrap()
        ),
    };
}
