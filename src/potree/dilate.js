

import {Timer} from "potree";

let shaderSource = `

	[[block]] struct Uniforms {
		uTest : u32;
		x : f32;
		y : f32;
		width : f32;
		height : f32;
		near : f32;
		window  : i32;
	};

	[[binding(0), group(0)]] var<uniform> uniforms   : Uniforms;
	[[binding(2), group(0)]] var myTexture           : texture_2d<f32>;
	[[binding(3), group(0)]] var myDepth             : texture_depth_2d;

	struct VertexInput {
		[[builtin(vertex_index)]] index : u32;
	};

	struct VertexOutput {
		[[builtin(position)]] position : vec4<f32>;
		[[location(0)]] uv : vec2<f32>;
	};

	struct FragmentInput {
		[[builtin(position)]] fragCoord : vec4<f32>;
		[[location(0)]] uv: vec2<f32>;
	};

	struct FragmentOutput{
		[[builtin(frag_depth)]] depth : f32;
		[[location(0)]] color : vec4<f32>;
	};

	fn toLinear(depth: f32, near: f32) -> f32{
		return near / depth;
	}

	fn getLinearDepthAt(fragment : FragmentInput, depthCoord : vec2<f32>) -> f32 {
		var currentCoord = fragment.fragCoord.xy;

		var pixelDepth = textureLoad(myDepth, vec2<i32>(depthCoord), 0);
		var diff = (currentCoord - depthCoord) / f32(uniforms.window);

		var depth = uniforms.near / pixelDepth;
		var wRadius = uniforms.near * f32(uniforms.window) * depth / f32(uniforms.width);
		var wi = diff.x * diff.x + diff.y * diff.y;
		var newDepth = depth + wi * wi * wRadius;

		return newDepth;
	}

	fn getWeight(fragment : FragmentInput, depthCoord : vec2<f32>) -> f32 {
		var currentCoord = fragment.fragCoord.xy;

		var pixelDepth = textureLoad(myDepth, vec2<i32>(depthCoord), 0);
		var diff = (currentCoord - depthCoord) / f32(uniforms.window);

		var depth = uniforms.near / pixelDepth;
		var wRadius = uniforms.near * f32(uniforms.window) * depth / f32(uniforms.width);
		var wi = diff.x * diff.x + diff.y * diff.y;
		var newDepth = depth + wi * wRadius * 0.5;

		return wi;
	}

	//=================================================
	//=================================================

	[[stage(vertex)]]
	fn main_vertex(vertex : VertexInput) -> VertexOutput {

		var pos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
			vec2<f32>(0.0, 0.0),
			vec2<f32>(0.1, 0.0),
			vec2<f32>(0.1, 0.1),
			vec2<f32>(0.0, 0.0),
			vec2<f32>(0.1, 0.1),
			vec2<f32>(0.0, 0.1)
		);

		var uv : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
			vec2<f32>(0.0, 1.0),
			vec2<f32>(1.0, 1.0),
			vec2<f32>(1.0, 0.0),
			vec2<f32>(0.0, 1.0),
			vec2<f32>(1.0, 0.0),
			vec2<f32>(0.0, 0.0)
		);

		var output : VertexOutput;

		output.position = vec4<f32>(pos[vertex.index], 0.999, 1.0);
		output.uv = uv[vertex.index];

		var x : f32 = uniforms.x * 2.0 - 1.0;
		var y : f32 = uniforms.y * 2.0 - 1.0;
		var width : f32 = uniforms.width * 2.0;
		var height : f32 = uniforms.height * 2.0;

		var vi = vertex.index;
		
		if(vi == 0u || vi == 3u || vi == 5u){
			output.position.x = x;
		}else{
			output.position.x = x + width;
		}

		if(vi == 0u || vi == 1u || vi == 3u){
			output.position.y = y;
		}else{
			output.position.y = y + height;
		}

		return output;
	}

	//=================================================
	//=================================================

	[[stage(fragment)]]
	fn main_fragment(input : FragmentInput) -> FragmentOutput {

		ignore(myTexture);
		ignore(myDepth);

		var window : i32 = uniforms.window;
		var DEFAULT_CLOSEST = 10000000.0;
		var closest = DEFAULT_CLOSEST;
		var closestCoords = vec2<f32>(0.0, 0.0);

		for(var i : i32 = -window; i <= window; i = i + 1){
		for(var j : i32 = -window; j <= window; j = j + 1){
			var coords : vec2<i32>;
			coords.x = i32(input.fragCoord.x) + i;
			coords.y = i32(input.fragCoord.y) + j;

			// var distance = sqrt(f32(i * i + j * j));

			var d : f32 = getLinearDepthAt(input, vec2<f32>(coords));

			if(d == 0.0){
				continue;
			}

			closest = min(closest, d);

			if(closest == d){
				closestCoords = input.fragCoord.xy + vec2<f32>(f32(i), f32(j));
			}
		}
		}

		var output : FragmentOutput;

		if(closest != DEFAULT_CLOSEST){

			var source = closestCoords;

			var color : vec4<f32> = textureLoad(myTexture, vec2<i32>(source), 0);
			var pixelDepth = textureLoad(myDepth, vec2<i32>(source), 0);
			var d = getLinearDepthAt(input, source);

			output.depth = pixelDepth;
			output.color = color;


			// var w = getWeight(input, source);
			// output.color = vec4<f32>(w, w, w, 1.0);

			return output;
		}else{
			discard;
		}

		return output;
	}
`;

let pipeline = null;
// let uniformBindGroup = null;
let uniformBuffer = null;

function init(renderer){

	if(pipeline !== null){
		return;
	}

	let {device, swapChainFormat} = renderer;

	let module = device.createShaderModule({code: shaderSource});

	pipeline = device.createRenderPipeline({
		vertex: {
			module: module,
			entryPoint: "main_vertex",
		},
		fragment: {
			module: module,
			entryPoint: "main_fragment",
			targets: [{format: "bgra8unorm"}],
		},
		primitive: {
			topology: 'triangle-list',
			cullMode: 'none',
		},
		depthStencil: {
				depthWriteEnabled: true,
				depthCompare: "greater",
				format: "depth32float",
		},
	});

	let uniformBufferSize = 256;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});
}

let uniformBindGroupCache = new Map();
function getUniformBindGroup(renderer, source){

	if(!uniformBindGroupCache.has(source)){
		
		let uniformBindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [
				{binding: 0, resource: {buffer: uniformBuffer}},
				{binding: 2, resource: source.colorAttachments[0].texture.createView()},
				{binding: 3, resource: source.depth.texture.createView({aspect: "depth-only"})}
			],
		});

		uniformBindGroupCache.set(source, uniformBindGroup);

	}

	return uniformBindGroupCache.get(source);
}

export function dilate(source, drawstate){

	let {renderer, camera, pass} = drawstate;
	let {passEncoder} = pass;

	init(renderer);

	Timer.timestamp(passEncoder,"dilate-start");

	let uniformBindGroup = getUniformBindGroup(renderer, source);

	passEncoder.setPipeline(pipeline);

	{ // update uniforms
		let source = new ArrayBuffer(32);
		let view = new DataView(source);

		let size = Potree.settings.pointSize;
		let window = Math.round((size - 1) / 2);

		view.setUint32(0, 5, true);
		view.setFloat32(4, 0, true);
		view.setFloat32(8, 0, true);
		view.setFloat32(12, 1, true);
		view.setFloat32(16, 1, true);
		view.setFloat32(20, camera.near, true);
		view.setInt32(24, window, true);
		
		renderer.device.queue.writeBuffer(
			uniformBuffer, 0,
			source, 0, source.byteLength
		);

		passEncoder.setBindGroup(0, uniformBindGroup);
	}

	passEncoder.draw(6, 1, 0, 0);

	Timer.timestamp(passEncoder,"dilate-end");

}