
import {Vector3, Matrix4, Geometry} from "potree";

const shaderSource = `
[[block]] struct Uniforms {
	view                : mat4x4<f32>;
	proj                : mat4x4<f32>;
	screen_width        : f32;
	screen_height       : f32;
	point_size          : f32;
	index               : u32;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var mySampler: sampler;
[[binding(2), group(0)]] var myTexture: texture_2d<f32>;

struct VertexIn{
	[[location(0)]] position : vec3<f32>;
	[[location(1)]] color    : vec3<f32>;
	[[location(2)]] uv       : vec2<f32>;
	[[builtin(vertex_index)]] vertexID : u32;
};

struct VertexOut{
	[[builtin(position)]] position : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
	[[location(1)]] uv : vec2<f32>;
};

struct FragmentIn{
	[[location(0)]] color : vec4<f32>;
	[[location(1)]] uv : vec2<f32>;
};

struct FragmentOut{
	[[location(0)]] color : vec4<f32>;
};


[[stage(vertex)]]
fn main_vertex(vertex : VertexIn) -> VertexOut {

	_ = vertex.uv;

	var projPos = uniforms.proj * uniforms.view * vec4<f32>(vertex.position, 1.0);

	var vout : VertexOut;
	vout.position = projPos;
	vout.color = vec4<f32>(vertex.color, 1.0);
	vout.uv = vertex.uv;


	return vout;
}

var<private> GRADIENT : array<vec3<f32>, 4> = array<vec3<f32>, 4>(
	vec3<f32>(215.0,  25.0,  28.0),
	vec3<f32>(253.0, 174.0,  97.0),
	vec3<f32>(171.0, 221.0, 164.0),
	vec3<f32>( 43.0, 131.0, 186.0),
);

[[stage(fragment)]]
fn main_fragment(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;
	// fout.color = vec4<f32>(fragment.uv, 0.0, 1.0);
	// fout.color = textureSample(myTexture, mySampler, fragment.uv);

	// {
	// 	var gIndex = uniforms.index % 4u;

	// 	fout.color = vec4<f32>(GRADIENT[gIndex] / 255.0, 1.0);
	// }

	_ = myTexture;
	_ = mySampler;

	return fout;
}
`;

let defaultTexture = null;

let sampler = null;
function getSampler(renderer){

	if(sampler){
		return sampler;
	}

	sampler = renderer.device.createSampler({
		magFilter: 'linear',
		minFilter: 'linear',
		mipmapFilter : 'linear',
		addressModeU: "repeat",
		addressModeV: "repeat",
		maxAnisotropy: 1,
	});

	return sampler;
}

function init(renderer){

	if(defaultTexture == null){
		let array = new ArrayBuffer(4);

		defaultTexture = renderer.createTextureFromArray(array, 1, 1);
	}

}

export function render(meshes, drawstate){

	if(meshes.length === 0){
		return;
	}

	let {renderer} = drawstate;
	let {device} = renderer;
	let {passEncoder} = drawstate.pass;

	init(renderer);

	let pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_vertex",
			buffers: [
				{ // position
					arrayStride: 3 * 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 0,
						offset: 0,
						format: "float32x3",
					}],
				},{ // color
					arrayStride: 4,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 1,
						offset: 0,
						format: "unorm8x4",
					}],
				},{ // uv
					arrayStride: 8,
					stepMode: "vertex",
					attributes: [{ 
						shaderLocation: 2,
						offset: 0,
						format: "float32x2",
					}],
				}
			]
		},
		fragment: {
			module: device.createShaderModule({code: shaderSource}),
			entryPoint: "main_fragment",
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

	passEncoder.setPipeline(pipeline);

	for(let batch of meshes){

		let uniformBuffer = device.createBuffer({
			size: 256,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		{ // update uniforms

			let {renderer, camera} = drawstate;

			let data = new ArrayBuffer(256);
			let f32 = new Float32Array(data);
			let view = new DataView(data);

			{ // transform

				// TODO: need set world on a per-mesh basis
				let world = meshes[0].world ?? new Matrix4();
				let view = camera.view;
				let worldView = new Matrix4().multiplyMatrices(view, world);

				f32.set(worldView.elements, 0);
				f32.set(camera.proj.elements, 16);
			}

			{ // misc
				let size = renderer.getSize();

				view.setFloat32(128, size.width, true);
				view.setFloat32(132, size.height, true);
				view.setFloat32(136, 5.0, true);
				view.setUint32(140, batch.index ?? 0, true);
			}

			renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
		}

		let vboPosition = batch.gpu_position ?? renderer.getGpuBuffer(batch.positions);
		passEncoder.setVertexBuffer(0, vboPosition);

		if(batch.gpu_color){
			passEncoder.setVertexBuffer(1, batch.gpu_color);
		}else if(batch.colors){
			let vboColor = renderer.getGpuBuffer(batch.colors);
			passEncoder.setVertexBuffer(1, vboColor);
		}else{
			passEncoder.setVertexBuffer(1, vboPosition);
		}
		

		let texture = defaultTexture;
		if(batch.image){
			texture = renderer.getGpuTexture(batch.image);
		}

		let bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0,resource: {buffer: uniformBuffer}},
				{binding: 1, resource: getSampler(renderer)},
				{binding: 2, resource: texture.createView()},
			],
		});

		passEncoder.setBindGroup(0, bindGroup);

		if(batch.gpu_uv){
			passEncoder.setVertexBuffer(2, batch.gpu_uv);
		}else if(batch.uvs){
			let vboUV = renderer.getGpuBuffer(batch.uvs);
			passEncoder.setVertexBuffer(2, vboUV);
		}else{
			passEncoder.setVertexBuffer(2, vboPosition);
		}

		if(batch.indices){

			let numIndices = batch.indices.length;
			let gpu_indices = renderer.getGpuBuffer(batch.indices);

			passEncoder.setIndexBuffer(gpu_indices, "uint32", 0, batch.indices.byteLength);

			passEncoder.drawIndexed(numIndices);

		}else if(batch.gpu_indices){

			let numIndices = batch.numIndices;
			let gpu_indices = batch.gpu_indices;

			passEncoder.setIndexBuffer(gpu_indices, "uint32", 
				4 * batch.firstIndex, 
				4 * numIndices
			);

			passEncoder.drawIndexed(numIndices);

		}else{
			let numVertices = batch.positions.length / 3;
			passEncoder.draw(numVertices, 1, 0, 0);
		}

		
	}

};