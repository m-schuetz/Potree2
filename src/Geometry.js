


export class Geometry{
	constructor(numPrimitives, buffers){
		this.numPrimitives = numPrimitives;
		this.buffers = buffers;
	}

	static createBox(){

		let position = new Float32Array([
			// bottom
			-1, -1, -1,    1, -1, -1,    1, 1, -1,
			-1, -1, -1,    1, 1, -1,    -1, 1, -1,

			// top
			-1, -1, 1,    1, -1, 1,    1, 1, 1,
			-1, -1, 1,    1, 1, 1,    -1, 1, 1,

			// back
			-1, 1, -1,    1, 1, -1,    1, 1, 1,
			-1, 1, -1,    1, 1, 1,    -1, 1, 1,

			// front
			-1, -1, -1,    1, -1, -1,    1, -1, 1,
			-1, -1, -1,    1, -1, 1,    -1, -1, 1,

			// left
			-1, -1, -1,    -1, 1, -1,    -1, 1, 1,
			-1, -1, -1,    -1, 1, 1,    -1, -1, 1,

			// right
			1, -1, -1,    1, 1, -1,    1, 1, 1,
			1, -1, -1,    1, 1, 1,    1, -1, 1,
		]);

		let colorValues = [];
		for(let i = 0; i < position.length; i += 3){
			let red = position[i + 0] < 0 ? 0 : 255;
			let green = position[i + 1] < 0 ? 0 : 255;
			let blue = position[i + 2] < 0 ? 0 : 255;

			colorValues.push(red, green, blue, 255);
		}
		let color = new Uint8Array(colorValues);

		let numTriangles = position.length / 3;

		let buffers = [
			{name: "position", array: position},
			{name: "color", array: color},
		];

		let geometry = new Geometry(numTriangles, buffers);

		return geometry;
	}
}

