


class Mesh{
	constructor(n, bufPositions, bufRGBA){
		this.n = n;
		this.bufPositions = bufPositions;
		this.bufRGBA = bufRGBA;
	}
}



export function createTestMesh(renderer){
	let n = 1_000_000;

	let {device} = renderer;

	let position = new Float32Array(3 * n);
	let color = new Uint8Array(4 * n);

	for(let i = 0; i < n; i++){
		let x = Math.random() - 0.5;
		let y = Math.random() - 0.5;
		let z = Math.random() - 0.5;

		let r = Math.random() * 255;
		let g = Math.random() * 255;
		let b = Math.random() * 255;

		position[3 * i + 0] = x;
		position[3 * i + 1] = y;
		position[3 * i + 2] = z;

		color[4 * i + 0] = r;
		color[4 * i + 1] = g;
		color[4 * i + 2] = b;
		color[4 * i + 3] = 255;
	}

	let [bufPositions, posMapping] = device.createBufferMapped({
		size: 12 * n,
		usage: GPUBufferUsage.VERTEX,
	});
	new Float32Array(posMapping).set(new Float32Array(position));
	bufPositions.unmap();

	let [bufRGBA, mappingRGB] = device.createBufferMapped({
		size: 4 * n,
		usage: GPUBufferUsage.VERTEX,
	});
	new Uint8Array(mappingRGB).set(new Uint8Array(color));
	bufRGBA.unmap();


	let mesh = new Mesh(n, bufPositions, bufRGBA);

	return mesh;
}



