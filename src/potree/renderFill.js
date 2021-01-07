
import {render as renderPoints}  from "./renderPoints.js";
import {RenderTarget} from "../core/RenderTarget.js";






let _target = null;

function getTarget(renderer){
	if(_target === null){

		let size = [128, 128, 1];
		_target = new RenderTarget(renderer, {
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

	return _target;
}







export function renderFill(renderer, pointcloud, camera){

	let target = getTarget(renderer);

	let renderPassDescriptor = {
		colorAttachments: [
			{
				attachment: target.colorAttachments[0].texture.createView(),
				loadValue: { r: 0.2, g: 0.4, b: 0.3, a: 1.0 },
			},
		],
		depthStencilAttachment: {
			attachment: target.depth.texture.createView(),
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

	return target;
}