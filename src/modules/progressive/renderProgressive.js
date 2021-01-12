
// import {drawQuads} from "../points/drawQuads.js";

// let renderTarget = null;

// function getRenderTarget(renderer){

// 	if(!renderTarget){
// 		renderTarget = new RenderTarget(renderer);

// 		let size = [128, 128, 1];
// 		renderTarget.size = [128, 128];

// 		{ // COLOR 0
// 			let descriptor = {
// 				size: size,
// 				format: renderer.swapChainFormat,
// 				usage: GPUTextureUsage.SAMPLED 
// 					| GPUTextureUsage.COPY_SRC 
// 					| GPUTextureUsage.COPY_DST 
// 					| GPUTextureUsage.OUTPUT_ATTACHMENT,
// 			};

// 			let texture = renderer.device.createTexture(descriptor);

// 			renderTarget.colorAttachments.push({descriptor, texture});
// 		}

// 		{ // COLOR 1
// 			let descriptor = {
// 				size: size,
// 				format: "r32uint",
// 				usage: GPUTextureUsage.SAMPLED 
// 					| GPUTextureUsage.COPY_SRC 
// 					| GPUTextureUsage.COPY_DST 
// 					| GPUTextureUsage.OUTPUT_ATTACHMENT,
// 			};

// 			let texture = renderer.device.createTexture(descriptor);

// 			renderTarget.colorAttachments.push({descriptor, texture});
// 		}

// 		{ // DEPTH
// 			let descriptor = {
// 				size: size,
// 				format: "depth24plus-stencil8",
// 				usage: GPUTextureUsage.OUTPUT_ATTACHMENT,
// 			};

// 			let texture = renderer.device.createTexture(descriptor);

// 			renderTarget.depth = {descriptor, texture};
// 		}
// 	}

// 	return renderTarget;
// }

// export function renderProgressive(renderer, points, camera){

// 	let renderTarget = getRenderTarget(renderer);
// 	renderTarget.setSize(800, 600);

	
// 	let renderPassDescriptor = {
// 		colorAttachments: [
// 			{
// 				attachment: renderTarget.colorAttachments[0].texture.createView(),
// 				loadValue: { r: 0.1, g: 0.3, b: 0.2, a: 1.0 },
// 			},
// 			// {
// 			// 	attachment: renderTarget.colorAttachments[1].texture.createView(),
// 			// 	loadValue: { r: 0.1, g: 0.3, b: 0.2, a: 1.0 },
// 			// },
// 		],
// 		depthStencilAttachment: {
// 			attachment: renderTarget.depth.texture.createView(),
// 			depthLoadValue: 1.0,
// 			depthStoreOp: "store",
// 			stencilLoadValue: 0,
// 			stencilStoreOp: "store",
// 		},
// 		sampleCount: 1,
// 	};

// 	const commandEncoder = renderer.device.createCommandEncoder();
// 	const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);

// 	let pass = {commandEncoder, passEncoder, renderPassDescriptor};
// 	drawQuads(renderer, pass, points, camera);

// 	passEncoder.endPass();

// 	renderer.device.defaultQueue.submit([commandEncoder.finish()]);



// 	return renderTarget;
// }


