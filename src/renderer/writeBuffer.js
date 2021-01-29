
let cs = `

[[block]] struct U32s {
	[[offset(0)]] values : [[stride(4)]] array<u32>;
};

[[block]] struct Uniforms {
	[[offset(0)]] offset : u32;
	[[offset(4)]] size : u32;
};
[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), set(0)]] var<storage_buffer> ssbo_source : U32s;
[[binding(2), set(0)]] var<storage_buffer> ssbo_target : U32s;

[[builtin(global_invocation_id)]] var<in> GlobalInvocationID : vec3<u32>;

fn readU8(offset : u32) -> u32{
	var ipos : u32 = offset / 4u;
	var val_u32 : u32 = ssbo_source.values[ipos];

	var shift : u32 = 8u * (offset % 4u);
	var val_u8 : u32 = (val_u32 >> shift) & 0xFFu;

	return val_u8;
}

[[stage(compute)]]
fn main() -> void {

	var index : u32 = GlobalInvocationID.x;

	var sourceIndex : u32 = index;
	var targetIndex : u32 = (4u * index + uniforms.offset) / 4u;

	var old : u32 = ssbo_target.values[index];
	var newValue : u32 = 0u;
	var shift : u32 = uniforms.offset % 4u;

	if(targetIndex == 0u){
		// first

		if(shift == 1u){
			newValue = old & 0xFFu;
			newValue = newValue | (readU8(0u) <<  8u);
			newValue = newValue | (readU8(1u) << 16u);
			newValue = newValue | (readU8(2u) << 24u);
		}elseif(shift == 2u){
			newValue = old & 0xFFFFu;
			newValue = newValue | (readU8(0u) << 16u);
			newValue = newValue | (readU8(1u) << 24u);
		}elseif(shift == 3u){
			newValue = old & 0xFFFFFFu;
			newValue = newValue | (readU8(0u) << 24u);
		}else{
			newValue = newValue | (readU8(0u) <<  0u);
			newValue = newValue | (readU8(1u) <<  8u);
			newValue = newValue | (readU8(2u) << 16u);
			newValue = newValue | (readU8(3u) << 24u);
		}

	}elseif(targetIndex == uniforms.size / 4u){
		// last, if overflow

		if(shift == 1u){
			newValue = old & 0xFFFFFF00u;
			newValue = newValue | (readU8(4u * sourceIndex + shift + 0u) << 0u);
		}elseif(shift == 2u){
			newValue = old >> 0xFFFF0000u;
			newValue = newValue | (readU8(4u * sourceIndex + shift + 0u) << 0u);
			newValue = newValue | (readU8(4u * sourceIndex + shift + 1u) << 8u);
		}elseif(shift == 3u){
			newValue = old >> 0xFF000000u;
			newValue = newValue | (readU8(4u * sourceIndex + shift + 0u) <<  0u);
			newValue = newValue | (readU8(4u * sourceIndex + shift + 1u) <<  8u);
			newValue = newValue | (readU8(4u * sourceIndex + shift + 2u) << 16u);
		}

	}else{
		// inner

		newValue = newValue | readU8(4u * sourceIndex + shift + 0u) <<  0u;
		newValue = newValue | readU8(4u * sourceIndex + shift + 1u) <<  8u;
		newValue = newValue | readU8(4u * sourceIndex + shift + 2u) << 16u;
		newValue = newValue | readU8(4u * sourceIndex + shift + 3u) << 24u;

	}

	ssbo_target.values[targetIndex] = newValue;

}

`;

let initialized = false;
let csModule;
let pipeline;
let uniformBuffer;

function init(renderer){

	if(initialized){
		return;
	}

	let {device} = renderer;

	csModule = device.createShaderModule({code: cs});
	
	pipeline = device.createComputePipeline({
		computeStage: {
			module: csModule,
			entryPoint: "main",
		}
	});

	uniformBuffer = device.createBuffer({
		size: 8,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	initialized = true;
}


export function writeBuffer(renderer, {target, targetOffset, source, sourceOffset, size}){
	let {device} = renderer;

	init(renderer);

	let size4 = size + 4 - (size % 4);
	let src = new Uint8Array(size4);
	src.set(new Uint8Array(source));
	
	let gpuSource = device.createBuffer({
		size: size4,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
	});
	device.queue.writeBuffer(
		gpuSource, 0,
		src.buffer, 0, src.byteLength
	);

	{ // uniform buffer

		let data = new Uint32Array([targetOffset, size]);
		device.queue.writeBuffer(
			uniformBuffer,
			0,
			data.buffer,
			data.byteOffset,
			data.byteLength
		);
	}

	const commandEncoder = device.createCommandEncoder();
	let passEncoder = commandEncoder.beginComputePass();

	let bindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{
				binding: 0,
				resource: {buffer: uniformBuffer}
			},{
				binding: 1,
				resource: {buffer: gpuSource}
			},{
				binding: 2,
				resource: {buffer: target}
			}
		]
	});

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	// let groups = Math.ceil(size / 128);
	// passEncoder.dispatch(groups, 1, 1);
	passEncoder.dispatch(Math.ceil(size) / 4);

	passEncoder.endPass();
	device.queue.submit([commandEncoder.finish()]);
}