
import {Vector3, Matrix4, Geometry} from "potree";

const vs = `
struct Uniforms {
	worldView      : mat4x4<f32>,
	proj           : mat4x4<f32>,
	screen_width   : f32,
	screen_height  : f32,
	size           : f32,
};

@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct VertexIn{
	@location(0) position : vec3<f32>,
	@location(1) color : vec3<f32>,
	@builtin(vertex_index) vertexID : u32,
};

struct VertexOut{
	@builtin(position) position : vec4<f32>,
	@location(0) color : vec4<f32>,
};

var<private> CUBE_POS : array<vec3<f32>, 36> = array<vec3<f32>, 36>(
	vec3<f32>(-0.5, -0.5, -0.5),
	vec3<f32>(0.5,  0.5, -0.5),
	vec3<f32>(0.5, -0.5, -0.5),
	vec3<f32>(-0.5, -0.5, -0.5),
	vec3<f32>(-0.5,  0.5, -0.5),
	vec3<f32>(0.5,  0.5, -0.5),
	vec3<f32>(-0.5, -0.5,  0.5),
	vec3<f32>(0.5, -0.5,  0.5),
	vec3<f32>(0.5,  0.5,  0.5),
	vec3<f32>(-0.5, -0.5,  0.5),
	vec3<f32>(0.5,  0.5,  0.5),
	vec3<f32>(-0.5,  0.5,  0.5),
	vec3<f32>(-0.5, -0.5, -0.5,),
	vec3<f32>(-0.5,  0.5,  0.5,),
	vec3<f32>(-0.5,  0.5, -0.5,),
	vec3<f32>(-0.5, -0.5, -0.5,),
	vec3<f32>(-0.5, -0.5,  0.5,),
	vec3<f32>(-0.5,  0.5,  0.5,),
	vec3<f32>(0.5, -0.5, -0.5),
	vec3<f32>(0.5,  0.5, -0.5),
	vec3<f32>(0.5,  0.5,  0.5),
	vec3<f32>(0.5, -0.5, -0.5),
	vec3<f32>(0.5,  0.5,  0.5),
	vec3<f32>(0.5, -0.5,  0.5),
	vec3<f32>(-0.5, 0.5, -0.5),
	vec3<f32>(0.5, 0.5,  0.5),
	vec3<f32>(0.5, 0.5, -0.5),
	vec3<f32>(-0.5, 0.5, -0.5),
	vec3<f32>(-0.5, 0.5,  0.5),
	vec3<f32>(0.5, 0.5,  0.5),
	vec3<f32>(-0.5, -0.5, -0.5),
	vec3<f32>(0.5, -0.5, -0.5),
	vec3<f32>(0.5, -0.5,  0.5),
	vec3<f32>(-0.5, -0.5, -0.5),
	vec3<f32>(0.5, -0.5,  0.5),
	vec3<f32>(-0.5, -0.5,  0.5),
);

@vertex
fn main(vertex : VertexIn) -> VertexOut {

	let cubeVertexIndex : u32 = vertex.vertexID % 36u;
	var cubeOffset : vec3<f32> = CUBE_POS[cubeVertexIndex];
	var voxelSize = uniforms.size;

	var viewPos : vec4<f32> = uniforms.worldView * vec4<f32>(vertex.position + voxelSize * cubeOffset, 1.0);
	var projPos : vec4<f32> = uniforms.proj * viewPos;

	var vout : VertexOut;
	vout.position = projPos;
	vout.color = vec4<f32>(vertex.color, 1.0);

	return vout;
}
`;

const fs = `

struct FragmentIn{
	@location(0) color : vec4<f32>,
};

struct FragmentOut{
	@location(0) color : vec4<f32>,
};

@fragment
fn main(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	return fout;
}
`;

let initialized = false;
let pipeline = null;
let uniformBuffer = null;
let bindGroup = null;

function createPipeline(renderer){

	let {device} = renderer;

	pipeline = device.createRenderPipeline({
		layout: "auto",
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
			buffers: [
				{ // position
					arrayStride: 3 * 4,
					stepMode: "instance",
					attributes: [{
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // color
					arrayStride: 4,
					stepMode: "instance",
					attributes: [{
						shaderLocation: 1,
						offset: 0,
						format: "unorm8x4",
					}],
				}
			]
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'back',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	return pipeline;
}

function init(renderer){

	if(initialized){
		return;
	}

	{
		pipeline = createPipeline(renderer);

		let {device} = renderer;
		// const uniformBufferSize = 256;

		// uniformBuffer = device.createBuffer({
		// 	size: uniformBufferSize,
		// 	usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		// });

		// bindGroup = device.createBindGroup({
		// 	layout: pipeline.getBindGroupLayout(0),
		// 	entries: [
		// 		{binding: 0,resource: {buffer: uniformBuffer}},
		// 	],
		// });
	}

	initialized = true;

}

let bindGroupCache = new Map();
function getBindGroup(renderer, batch){

	if(!bindGroupCache.has(batch)){
		const uniformBufferSize = 256;

		let arrayBuffer = new ArrayBuffer(uniformBufferSize);

		let uniformBuffer = renderer.device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		let bindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0,resource: {buffer: uniformBuffer}},
			],
		});

		let uniformData = {arrayBuffer, uniformBuffer, bindGroup};
		bindGroupCache.set(batch, uniformData);
	}

	return bindGroupCache.get(batch);
}

export function render(batches, drawstate){

	let {renderer, camera} = drawstate;
	let {device} = renderer;

	init(renderer);

	// updateUniforms(drawstate);

	let {passEncoder} = drawstate.pass;

	passEncoder.setPipeline(pipeline);

	for(let batch of batches){

		let unfiformData = getBindGroup(renderer, batch.positions);

		{
			

			passEncoder.setBindGroup(0, unfiformData.bindGroup);

			let data = unfiformData.arrayBuffer;
			let f32 = new Float32Array(data);
			let view = new DataView(data);

			{ // transform
				let world = new Matrix4();
				let view = camera.view;
				let worldView = new Matrix4().multiplyMatrices(view, world);

				f32.set(worldView.elements, 0);
				f32.set(camera.proj.elements, 16);
			}

			{ // misc
				let size = renderer.getSize();

				view.setFloat32(128, size.width, true);
				view.setFloat32(132, size.height, true);
				view.setFloat32(136, batch.voxelSize, true);
			}

			renderer.device.queue.writeBuffer(unfiformData.uniformBuffer, 0, data, 0, data.byteLength);
		}

	


		let vboPosition = renderer.getGpuBuffer(batch.positions);
		let vboColor = renderer.getGpuBuffer(batch.colors);

		passEncoder.setVertexBuffer(0, vboPosition);
		passEncoder.setVertexBuffer(1, vboColor);

		let numVertices = batch.positions.length / 3;
		passEncoder.draw(36, numVertices, 0, 0);
	}

};