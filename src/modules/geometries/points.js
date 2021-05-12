
import {Geometry, Points} from "potree";

const {sin, cos} = Math;

export function createPointsData(cells = 1_000){
	let n = cells * cells;
	let position = new Float32Array(3 * n);
	let color = new Uint8Array(4 * n);
	let k = 0;
	for(let i = 0; i < cells; i++){
	for(let j = 0; j < cells; j++){

		let u = 2.0 * (i / cells) - 1.0;
		let v = 2.0 * (j / cells) - 1.0;

		let x = 2 * u;
		let y = 2 * v;
		let z = sin(3 * u) * cos(3 * v);

		position[3 * k + 0] = x;
		position[3 * k + 1] = y;
		position[3 * k + 2] = z;

		color[4 * k + 0] = 255 * u;
		color[4 * k + 1] = 255 * v;
		color[4 * k + 2] = 0;
		color[4 * k + 3] = 255;

		k++;
	}
	}


	let numElements = n;
	let buffers = [
		{
			name: "position",
			buffer: position,
		},{
			name: "rgba",
			buffer: color,
		}
	];
	let geometry = new Geometry({numElements, buffers});
	let points = new Points();
	points.geometry = geometry;

	return points;
}

export function createPointsSphere(n = 1_000){

	let position = new Float32Array(3 * n);
	let color = new Uint8Array(4 * n);
	for(let i = 0; i < n; i++){

		let x = 2 * Math.random() - 1;
		let y = 2 * Math.random() - 1;
		let z = 2 * Math.random() - 1;

		let d = Math.sqrt(x ** 2 + y ** 2 + z ** 2);
		x = x / d;
		y = y / d;
		z = z / d;

		position[3 * i + 0] = x;
		position[3 * i + 1] = y;
		position[3 * i + 2] = z;

		color[4 * i + 0] = 255 * (z / 2 + 0.5);
		color[4 * i + 1] = 0;
		color[4 * i + 2] = 0;
		color[4 * i + 3] = 255;
	}


	let numElements = n;
	let buffers = [
		{
			name: "position",
			buffer: position,
		},{
			name: "rgba",
			buffer: color,
		}
	];
	let geometry = new Geometry({numElements, buffers});
	let points = new Points();
	points.geometry = geometry;

	return points;
}