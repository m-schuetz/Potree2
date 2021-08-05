
import {Geometry, Vector3, Matrix4} from "potree";

const vs = `
[[block]] struct Uniforms {
	worldView : mat4x4<f32>;
	proj : mat4x4<f32>;
	screen_width : f32;
	screen_height : f32;
};

[[block]] struct U32s {
	values : [[stride(4)]] array<u32>;
};

[[block]] struct F32s {
	values : [[stride(4)]] array<f32>;
};

[[binding(0), group(0)]] var<uniform> uniforms : Uniforms;
[[binding(1), group(0)]] var<storage, read> positions : F32s;
[[binding(2), group(0)]] var<storage, read> colors : U32s;

struct VertexIn{
	[[builtin(vertex_index)]] vertexID : u32;
};

struct VertexOut{
	[[builtin(position)]]   position  : vec4<f32>;
	[[location(0)]]         color     : vec4<f32>;
};

fn loadVertex(index : u32) -> vec4<f32> {
	var position = vec4<f32>(
		positions.values[3u * index + 0u],
		positions.values[3u * index + 1u],
		positions.values[3u * index + 2u],
		1.0
	);

	return position;
}

fn toScreen(worldPos : vec4<f32>) -> vec2<f32> {

	var projPos = uniforms.proj * uniforms.worldView * worldPos;
	var lineWidth = 10.0;

	var fx : f32 = projPos.x / projPos.w;
	var fy : f32 = projPos.y / projPos.w;

	var screenPos = vec2<f32>(fx, fy);

	return screenPos;
}

[[stage(vertex)]]
fn main(vertex : VertexIn) -> VertexOut {

	// A line is made of 2 triangles / 6 vertices
	// each of the 6 vertices loads start and end of the line
	// and then places itself according to the local index
	var lineID = vertex.vertexID / 6u;
	var start = loadVertex(2u * lineID + 0u);
	var end = loadVertex(2u * lineID + 1u);
	var localIndex = vertex.vertexID % 6u;

	var position = start;
	if(localIndex == 0u || localIndex == 3u|| localIndex == 5u){
		position = start;
	}else{
		position = end;
	}

	var bColor = colors.values[2u * lineID];
	var color = vec4<f32>(
		f32((bColor >>  0u) & 0xFFu),
		f32((bColor >>  8u) & 0xFFu),
		f32((bColor >> 16u) & 0xFFu),
		1.0
	);

	var vout : VertexOut;
	
	var worldPos = position;
	var viewPos = uniforms.worldView * worldPos;
	var projPos = uniforms.proj * viewPos;

	var dirScreen : vec2<f32>;
	{
		var projStart = uniforms.proj * uniforms.worldView * start;
		var projEnd = uniforms.proj * uniforms.worldView * end;

		var screenStart = projStart.xy / projStart.w;
		var screenEnd = projEnd.xy / projEnd.w;

		dirScreen = normalize(screenEnd - screenStart);
	}

	{ // apply pixel offsets to the 6 vertices of the quad

		var lineWidth = 3.0;
		var pxOffset = vec2<f32>(1.0, 0.0);

		// move vertices of quad sidewards
		if(localIndex == 0u || localIndex == 1u || localIndex == 3u){
			pxOffset = vec2<f32>(dirScreen.y, -dirScreen.x);
		}else{
			pxOffset = vec2<f32>(-dirScreen.y, dirScreen.x);
		}

		// move vertices of quad outwards
		if(localIndex == 0u || localIndex == 3u || localIndex == 5u){
			pxOffset = pxOffset - dirScreen;
		}else{
			pxOffset = pxOffset + dirScreen;
		}

		var screenDimensions = vec2<f32>(uniforms.screen_width, uniforms.screen_height);
		var adjusted = projPos.xy / projPos.w + lineWidth * pxOffset / screenDimensions;
		projPos = vec4<f32>(adjusted * projPos.w, projPos.zw);
	}

	vout.position = projPos;

	// if(position.z < 0.0){
	// 	color = vec4<f32>(237.0 / 255.0 , 248.0 / 255.0 , 177.0 / 255.0, 1.0);
	// }else{
	// 	color = vec4<f32>(22.0 / 255.0 , 127.0 / 255.0 , 184.0 / 255.0, 1.0);
	// }

	vout.color = color;

	return vout;
}
`;

const fs = `

struct FragmentIn{
	[[builtin(position)]] position  : vec4<f32>;
	[[location(0)]] color : vec4<f32>;
};

struct FragmentOut{
	[[location(0)]] color : vec4<f32>;
	[[builtin(frag_depth)]] depth : f32;
};

[[stage(fragment)]]
fn main(fragment : FragmentIn) -> FragmentOut {

	var fout : FragmentOut;
	fout.color = fragment.color;

	fout.depth = fragment.position.z * 1.002;

	return fout;
}
`;


let vbo_lines = null;
let initialized = false;
let pipeline = null;
let uniformBuffer = null;
let bindGroup = null;
let capacity = 1_000_000;

function createPipeline(renderer){

	let {device} = renderer;

	pipeline = device.createRenderPipeline({
		vertex: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
			buffers: []
		},
		fragment: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
			depthWriteEnabled: true,
			depthCompare: 'greater',
			format: "depth32float",
		},
	});

	return pipeline;
}

function createBuffer(renderer, data){

	let {device} = renderer;

	let vbos = [];

	for(let entry of data.geometry.buffers){
		let {name, buffer} = entry;

		let vbo = device.createBuffer({
			size: buffer.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true,
		});

		let type = buffer.constructor;
		new type(vbo.getMappedRange()).set(buffer);
		vbo.unmap();

		vbos.push({
			name: name,
			vbo: vbo,
		});
	}

	return vbos;
}

function init(renderer){

	if(initialized){
		return;
	}

	{ // create lines vbo

		let geometry = {
			buffers: [{
				name: "position",
				buffer: new Float32Array(2 * 3 * capacity),
			},{
				name: "color",
				buffer: new Uint8Array(2 * 4 * capacity),
			}]
		};
		let node = {geometry};

		vbo_lines = createBuffer(renderer, node);
	}

	{
		pipeline = createPipeline(renderer);

		let {device} = renderer;
		const uniformBufferSize = 256;

		uniformBuffer = device.createBuffer({
			size: uniformBufferSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		bindGroup = device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 1, resource: {buffer: vbo_lines[0].vbo}},
				{binding: 2, resource: {buffer: vbo_lines[1].vbo}},
			],
		});
	}

	initialized = true;

}

function updateUniforms(drawstate){

	let {renderer, camera} = drawstate;

	let data = new ArrayBuffer(256);
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
	}

	renderer.device.queue.writeBuffer(uniformBuffer, 0, data, 0, data.byteLength);
}

export function render(lines, drawstate){

	let {renderer} = drawstate;
	let {device} = renderer;

	init(renderer);

	updateUniforms(drawstate);

	let {passEncoder} = drawstate.pass;

	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);

	{
		let position = new Float32Array(2 * 3 * lines.length);
		let color = new Uint8Array(2 * 4 * lines.length);

		for(let i = 0; i < lines.length; i++){
			let [start, end, c] = lines[i];

			position[6 * i + 0] = start.x;
			position[6 * i + 1] = start.y;
			position[6 * i + 2] = start.z;

			position[6 * i + 3] = end.x;
			position[6 * i + 4] = end.y;
			position[6 * i + 5] = end.z;

			color[8 * i + 0] = c.x;
			color[8 * i + 1] = c.y;
			color[8 * i + 2] = c.z;
			color[8 * i + 3] = 255;

			color[8 * i + 4] = c.x;
			color[8 * i + 5] = c.y;
			color[8 * i + 6] = c.z;
			color[8 * i + 7] = 255;
		}

		device.queue.writeBuffer(
			vbo_lines[0].vbo, 0,
			position.buffer, position.byteOffset, position.byteLength
		);

		device.queue.writeBuffer(
			vbo_lines[1].vbo, 0,
			color.buffer, color.byteOffset, color.byteLength
		);
	}

	// passEncoder.setVertexBuffer(0, vbo_lines[0].vbo);
	// passEncoder.setVertexBuffer(1, vbo_lines[1].vbo);

	passEncoder.draw(6 * lines.length, 1, 0, 0);

};