
let cs = `

struct Uniforms {
	numElements : u32;
	value       : u32;
};

struct U32s { values : array<u32> };

@binding(0) @group(0) var<uniform> uniforms : Uniforms;
@binding(1) @group(0) var<storage, read_write> target : U32s;

[[stage(compute), workgroup_size(128)]]
fn main([[builtin(global_invocation_id)]] GlobalInvocationID : vec3<u32>) {

	var index = GlobalInvocationID.x;

	if(index > uniforms.numElements){
		return;
	}

	target.values[index] = uniforms.value;
}
`;


let pipeline = null;
let uniformBuffer = null;

function init(renderer){

	if(pipeline !== null){
		return;
	}

	let {device} = renderer;

	pipeline =  device.createComputePipeline({
		compute: {
			module: device.createShaderModule({code: cs}),
			entryPoint: 'main',
		},
	});

}

export function fillBuffer(renderer, buffer, value, numU32Elements){
	init(renderer);

	let {device} = renderer;

	let uniform_flags = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;
	let uniformBuffer = device.createBuffer({size: 256, usage: uniform_flags});

	let bindGroup = renderer.device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [
			{binding: 0, resource: {buffer: uniformBuffer}},
			{binding: 1, resource: {buffer: buffer}}
		],
	});

	const commandEncoder = device.createCommandEncoder();
	const passEncoder = commandEncoder.beginComputePass();

	{ // update uniforms
		let source = new ArrayBuffer(256);
		let view = new DataView(source);

		view.setUint32(0, numU32Elements, true);
		view.setUint32(4, value, true);
		
		renderer.device.queue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);
	}

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	let groups = Math.ceil(numU32Elements / 128);
	passEncoder.dispatch(groups);
	passEncoder.endPass();
	
	device.queue.submit([commandEncoder.finish()]);
}