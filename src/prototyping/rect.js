
let vs = `
	let pos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(0.1, 0.0),
		vec2<f32>(0.1, 0.1),
		vec2<f32>(0.0, 0.0),
		vec2<f32>(0.1, 0.1),
		vec2<f32>(0.0, 0.1)
	);

	let uv : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(0.0, 0.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(0.0, 1.0)
	);

	[[builtin(position)]] var<out> Position : vec4<f32>;
	[[builtin(vertex_idx)]] var<in> VertexIndex : i32;

	[[block]] struct Uniforms {
		[[offset(0)]] uTest : u32;
		[[offset(4)]] x : f32;
		[[offset(8)]] y : f32;
		[[offset(12)]] width : f32;
		[[offset(16)]] height : f32;
	};
	[[binding(0), set(0)]] var<uniform> uniforms : Uniforms;

	[[location(0)]] var<out> fragUV : vec2<f32>;

	[[stage(vertex)]]
	fn main() -> void {
		Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
		fragUV = uv[VertexIndex];

		if(VertexIndex == 0){
			Position.x = uniforms.x;
			Position.y = uniforms.y;
		}elseif(VertexIndex == 1){
			Position.x = uniforms.x + uniforms.width;
			Position.y = uniforms.y;
		}elseif(VertexIndex == 2){
			Position.x = uniforms.x + uniforms.width;
			Position.y = uniforms.y + uniforms.height;
		}elseif(VertexIndex == 3){
			Position.x = uniforms.x;
			Position.y = uniforms.y;
		}elseif(VertexIndex == 4){
			Position.x = uniforms.x + uniforms.width;;
			Position.y = uniforms.y + uniforms.height;
		}elseif(VertexIndex == 5){
			Position.x = uniforms.x;
			Position.y = uniforms.y + uniforms.height;
		}

		# if(uniforms.uTest == 5){
		# 	fragUV = vec2<f32>(0.0, 1.0);
		# }

		return;
	}
`;

let fs = `
	[[location(0)]] var<out> outColor : vec4<f32>;

	[[location(0)]] var<in> fragUV: vec2<f32>;

	[[stage(fragment)]]
	fn main() -> void {

		#outColor = vec4<f32>(1.0, 0.0, 0.0, 1.0);
		outColor = vec4<f32>(fragUV, 0.0, 1.0);

		return;
	}
`;


let bindGroupLayout = null;
let pipeline = null;
let uniformBindGroup = null;
let uniformBuffer = null;

function getBindGroupLayout(renderer){

	if(!bindGroupLayout){
		let {device} = renderer;

		bindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				type: "uniform-buffer"
			}]
		});
	}

	return bindGroupLayout;
}

function getPipeline(renderer){

	if(pipeline){
		return pipeline;
	}

	let {device, swapChainFormat} = renderer;

	let bindGroupLayout = getBindGroupLayout(renderer);
	let layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

	pipeline = device.createRenderPipeline({
		layout: layout, 
		vertexStage: {
			module: device.createShaderModule({code: vs}),
			entryPoint: "main",
		},
		fragmentStage: {
			module: device.createShaderModule({code: fs}),
			entryPoint: "main",
		},
		primitiveTopology: "triangle-list",
		depthStencilState: {
				depthWriteEnabled: true,
				depthCompare: 'greater',
				format: "depth32float",
		},
		colorStates: [{
			format: swapChainFormat,
		}],
	});

	let uniformBufferSize = 24;
	uniformBuffer = device.createBuffer({
		size: uniformBufferSize,
		usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
	});

	uniformBindGroup = device.createBindGroup({
		layout: pipeline.getBindGroupLayout(0),
		entries: [{
			binding: 0,
			resource: {
				buffer: uniformBuffer,
			}
		}],
	});

	return pipeline;
}

export function drawRect(renderer, pass, x, y, width, height){

	let {device} = renderer;

	let pipeline = getPipeline(renderer);

	let source = new ArrayBuffer(24);
	let view = new DataView(source);
	view.setUint32(0, 5, true);
	view.setFloat32(4, x, true);
	view.setFloat32(8, y, true);
	view.setFloat32(12, width, true);
	view.setFloat32(16, height, true);
	device.queue.writeBuffer(
		uniformBuffer, 0,
		source, 0, source.byteLength
	);

	let {passEncoder} = pass;
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, uniformBindGroup);
	passEncoder.draw(6, 1, 0, 0);

}