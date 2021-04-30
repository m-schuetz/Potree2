
export class RenderTarget{

	constructor(renderer, params = {}){
		this.colorAttachments = [];
		this.depth = null;
		this.size = params.size ?? [128, 128];
		this.renderer = renderer;
		this.version = 0;

		{ // COLOR ATTACHMENTS
			let descriptors = params.colorDescriptors ?? [{
				size: this.size,
				format: "r32uint",
				usage: GPUTextureUsage.SAMPLED 
					| GPUTextureUsage.COPY_SRC 
					| GPUTextureUsage.COPY_DST 
					| GPUTextureUsage.RENDER_ATTACHMENT,
			}];

			for(let descriptor of descriptors){
				let texture = renderer.device.createTexture(descriptor);

				this.colorAttachments.push({descriptor, texture});
			}
		}

		{ // DEPTH ATTACHMENT
			let descriptor = params.depthDescriptor ?? {
				size: this.size,
				format: "depth32float",
				usage: GPUTextureUsage.RENDER_ATTACHMENT
					| GPUTextureUsage.COPY_SRC,
			};

			let texture = renderer.device.createTexture(descriptor);

			this.depth = {descriptor, texture};
		}
	}

	// static create(descriptor){
	// 	let instance = Object.create(RenderTarget);

	// 	return instance;
	// }

	setSize(width, height){

		let resized = this.size[0] !== width || this.size[1] !== height;

		if(resized){

			this.size = [width, height];
			
			// resize color attachments
			for(let attachment of this.colorAttachments){
				attachment.texture.destroy();

				let desc = attachment.descriptor;
				desc.size = [...this.size, 1];

				attachment.texture = this.renderer.device.createTexture(desc);
			}

			{ // resize depth attachment
				let attachment = this.depth;
				attachment.texture.destroy();
				
				let desc = attachment.descriptor;
				desc.size = [...this.size, 1];

				attachment.texture = this.renderer.device.createTexture(desc);
			}

		}

	}

}