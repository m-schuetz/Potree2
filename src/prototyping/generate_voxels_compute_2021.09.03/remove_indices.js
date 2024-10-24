
import { Vector3 } from "../../math/Vector3.js";
import {Chunk, voxelGridSize, toIndex1D, toIndex3D, computeChildBox} from "./common.js";
import {storage_flags, uniform_flags} from "./common.js";
import { transferTriangles } from "./transferTriangles.js";
import {Geometry, Mesh, Box3} from "potree";

export let csChunking = `

struct Uniforms {
	chunkGridSize      : u32;
	pad_1              : u32;
	pad_0              : u32;
	batchIndex         : u32;
	boxMin             : vec4<f32>;      // offset(16)
	boxMax             : vec4<f32>;      // offset(32)
};

struct Batch {
	numTriangles      : u32;
	firstTriangle     : u32;
	lutCounter        : atomic<u32>;
};

struct F32s { values : array<f32>; };
struct U32s { values : array<u32> };
struct I32s { values : array<i32>; };
struct AU32s { values : array<atomic<u32>>; };
struct AI32s { values : array<atomic<i32>>; };
struct Batches { values : array<Batch>; };

[[binding( 0), group(0)]] var<uniform> uniforms : Uniforms;

[[binding(10), group(0)]] var<storage, read_write> indices   : U32s;
[[binding(11), group(0)]] var<storage, read_write> positions : F32s;
[[binding(12), group(0)]] var<storage, read_write> colors    : U32s;

[[binding(50), group(0)]] var<storage, read_write> sortedTriangles : U32s;
[[binding(50), group(0)]] var<storage, read_write> sortedTriangles : U32s;


[[stage(compute), workgroup_size(128)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	doIgnore();

	let batch = &batches.values[uniforms.batchIndex];

	if(GlobalInvocationID.x >= (*batch).numTriangles){
		return;
	}

}


`;

const maxBatchSize = 10_000_000;

export async function doChunking(renderer, mesh){




}