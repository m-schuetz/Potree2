
// export async function loadTexture(url){
// 	let img = document.createElement('img');
	
// 	img.src = url;
// 	await img.decode();

// 	let imageBitmap = await createImageBitmap(img);

// 	return imageBitmap;
// }

// let state = new Map();

// export function drawTexture(texture, renderer){

// 	let gpuTexture = device.createTexture({
// 		size: [texture.width, texture.height, 1],
// 		format: "rgba8unorm",
// 		usage: GPUTextureUsage.SAMPLED | GPUTextureUsage.COPY_DST,
// 	});

// 	device.defaultQueue.copyImageBitmapToTexture(
// 		{texture}, {texture: gpuTexture},
// 		[texture.width, texture.height, 1]
// 	);

// }