
import {render as renderPoints}  from "./renderPoints.js";
import {RenderTarget} from "../core/RenderTarget.js";



let vs = `
	const pos : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 0.0),
		vec2<f32>(0.1, 0.0),
		vec2<f32>(0.1, 0.1),
		vec2<f32>(0.0, 0.0),
		vec2<f32>(0.1, 0.1),
		vec2<f32>(0.0, 0.1)
	);

	const uv : array<vec2<f32>, 6> = array<vec2<f32>, 6>(
		vec2<f32>(0.0, 1.0),
		vec2<f32>(1.0, 1.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(0.0, 1.0),
		vec2<f32>(1.0, 0.0),
		vec2<f32>(0.0, 0.0)
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

		var x : f32 = uniforms.x * 2.0 - 1.0;
		var y : f32 = uniforms.y * 2.0 - 1.0;
		var width : f32 = uniforms.width * 2.0;
		var height : f32 = uniforms.height * 2.0;

		if(VertexIndex == 0){
			Position.x = x;
			Position.y = y;
		}elseif(VertexIndex == 1){
			Position.x = x + width;
			Position.y = y;
		}elseif(VertexIndex == 2){
			Position.x = x + width;
			Position.y = y + height;
		}elseif(VertexIndex == 3){
			Position.x = x;
			Position.y = y;
		}elseif(VertexIndex == 4){
			Position.x = x + width;
			Position.y = y + height;
		}elseif(VertexIndex == 5){
			Position.x = x;
			Position.y = y + height;
		}

		return;
	}
`;

let fs = `

	[[binding(1), set(0)]] var<uniform_constant> mySampler: sampler;
	[[binding(2), set(0)]] var<uniform_constant> myTexture: texture_sampled_2d<f32>;
	[[binding(3), set(0)]] var<uniform_constant> myDepth: texture_sampled_2d<f32>;
	#[[binding(3), set(0)]] var<uniform_constant> myDepth: texture_depth_2d;

	[[location(0)]] var<out> outColor : vec4<f32>;

	[[location(0)]] var<in> fragUV: vec2<f32>;

	[[builtin(frag_coord)]] var<in> fragCoord : vec4<f32>;

	[[stage(fragment)]]
	fn main() -> void {

		var avg : vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);

		var window : i32 = 2;
		var closest : f32 = 1.0;
		var far : f32 = 10000.0;

		for(var i : i32 = -window; i <= window; i = i + 1){
			for(var j : i32 = -window; j <= window; j = j + 1){
				var coords : vec2<i32>;
				coords.x = i32(fragCoord.x) + i;
				coords.y = i32(fragCoord.y) + j;

				var d : f32 = textureLoad(myDepth, coords).x;
				var linearDistance : f32 = d * far;

				closest = min(closest, d);
			}
		}

		
		for(var i : i32 = -window; i <= window; i = i + 1){
			for(var j : i32 = -window; j <= window; j = j + 1){
				var coords : vec2<i32>;
				coords.x = i32(fragCoord.x) + i;
				coords.y = i32(fragCoord.y) + j;

				var d : f32 = textureLoad(myDepth, coords).x;
				var linearDistance : f32 = d * far;


				if(d < closest * 1.01){
					var manhattanDistance : f32 = f32(abs(i) + abs(j));

					var weight : f32 = 1.0;

					if(manhattanDistance <= 0.0){
						weight = 1.0;
					}elseif(manhattanDistance <= 1.0){
						weight = 0.3;
					}elseif(manhattanDistance <= 2.0){
						weight = 0.01;
					}else{
						weight = 0.001;
					}
					
					var color : vec4<f32> = textureLoad(myTexture, coords);
					color.x = color.x * weight;
					color.y = color.y * weight;
					color.z = color.z * weight;
					color.w = color.w * weight;

					avg = avg + color;
				}
			}
		}

		if(avg.x + avg.y + avg.z == 0.0){
			outColor = vec4<f32>(0.1, 0.2, 0.3, 1.0);
		}else{
			avg.x = avg.x / avg.w;
			avg.y = avg.y / avg.w;
			avg.z = avg.z / avg.w;
			avg.w = 1.0;
			outColor = avg;
		}

		#outColor =  textureSample(myTexture, mySampler, fragUV);

		#outColor.x = fragCoord.x / 1200.0;
		#outColor.y = fragCoord.y / 800.0;

		return;
	}
`;

let bindGroupLayout = null;
let pipeline = null;
let uniformBindGroup = null;
let uniformBuffer = null;

let _target_1 = null;
let _target_2 = null;

function getTarget1(renderer){
	if(_target_1 === null){

		let size = [128, 128, 1];
		_target_1 = new RenderTarget(renderer, {
			size: size,
			colorDescriptors: [{
				size: size,
				format: renderer.swapChainFormat,
				usage: GPUTextureUsage.SAMPLED 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.OUTPUT_ATTACHMENT,
			}],
			depthDescriptor: {
				size: size,
				format: "depth24plus-stencil8",
				usage: GPUTextureUsage.SAMPLED 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.OUTPUT_ATTACHMENT,
			}
		});
	}

	return _target_1;
}

function getTarget2(renderer){
	if(_target_2 === null){

		let size = [128, 128, 1];
		_target_2 = new RenderTarget(renderer, {
			size: size,
			colorDescriptors: [{
				size: size,
				format: renderer.swapChainFormat,
				usage: GPUTextureUsage.SAMPLED 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.OUTPUT_ATTACHMENT,
			}],
			depthDescriptor: {
				size: size,
				format: "depth32float",
				usage: GPUTextureUsage.SAMPLED 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.OUTPUT_ATTACHMENT,
			}
		});
	}

	return _target_2;
}

function init(renderer){

	let target1 = getTarget1(renderer);
	let target2 = getTarget2(renderer);

	if(pipeline === null){
		let {device, swapChainFormat} = renderer;

		bindGroupLayout = device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				type: "uniform-buffer"
			},{
				// Sampler
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				type: "sampler"
			},{
				// Texture view
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				type: "sampled-texture"
			}
			,{
				// Texture view
				binding: 3,
				visibility: GPUShaderStage.FRAGMENT,
				type: "sampled-texture"
			}
			]
		});


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
					depthCompare: "less",
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

	}


}

export function renderFill(renderer, pointcloud, camera){

	init(renderer);

	let size = renderer.getSize();

	{ // PASS 1
		let target = getTarget1(renderer);
		target.setSize(size.width, size.height);

		let sampler = renderer.device.createSampler({
			magFilter: "linear",
			minFilter: "linear",
		});

		// TODO: possible issue: re-creating bind group every frame
		// doing that because the render target attachments may change after resize
		uniformBindGroup = renderer.device.createBindGroup({
			layout: pipeline.getBindGroupLayout(0),
			entries: [{
					binding: 0,
					resource: {
						buffer: uniformBuffer,
					}
				},{
					binding: 1,
					resource: sampler,
				},{
					binding: 2,
					resource: target.colorAttachments[0].texture.createView(),
				}
				,{
					binding: 3,
					resource: target.depth.texture.createView({aspect: "depth-only"}),
				}
			],
		});

		let renderPassDescriptor = {
			colorAttachments: [
				{
					attachment: target.colorAttachments[0].texture.createView(),
					loadValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
				},
			],
			depthStencilAttachment: {
				attachment: target.depth.texture.createView({aspect: "depth-only"}),
				depthLoadValue: 1.0,
				depthStoreOp: "store",
				stencilLoadValue: 0,
				stencilStoreOp: "store",
			},
			sampleCount: 1,
		};

		const commandEncoder = renderer.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		let pass = {commandEncoder, passEncoder, renderPassDescriptor};

		renderPoints(renderer, pass, pointcloud, camera);

		pass.passEncoder.endPass();
		let commandBuffer = pass.commandEncoder.finish();
		renderer.device.defaultQueue.submit([commandBuffer]);
	}

	{ // PASS 2

		let target = getTarget2(renderer);
		target.setSize(size.width, size.height);

		let renderPassDescriptor = {
			colorAttachments: [
				{
					attachment: target.colorAttachments[0].texture.createView(),
					loadValue: { r: 0.4, g: 0.2, b: 0.3, a: 1.0 },
				},
			],
			depthStencilAttachment: {
				attachment: target.depth.texture.createView({aspect: "depth-only"}),
				depthLoadValue: 1.0,
				depthStoreOp: "store",
				stencilLoadValue: 0,
				stencilStoreOp: "store",
			},
			sampleCount: 1,
		};

		const commandEncoder = renderer.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

		let pass = {commandEncoder, passEncoder, renderPassDescriptor};

		passEncoder.setPipeline(pipeline);

		{
			let source = new ArrayBuffer(24);
			let view = new DataView(source);

			view.setUint32(0, 5, true);
			view.setFloat32(4, 0, true);
			view.setFloat32(8, 0, true);
			view.setFloat32(12, 1, true);
			view.setFloat32(16, 1, true);
			
			renderer.device.defaultQueue.writeBuffer(
				uniformBuffer, 0,
				source, 0, source.byteLength
			);

			passEncoder.setBindGroup(0, uniformBindGroup);
		}

		passEncoder.draw(6, 1, 0, 0);


		// renderPoints(renderer, pass, pointcloud, camera);

		pass.passEncoder.endPass();
		let commandBuffer = pass.commandEncoder.finish();
		renderer.device.defaultQueue.submit([commandBuffer]);

	}

	return getTarget2(renderer);
}