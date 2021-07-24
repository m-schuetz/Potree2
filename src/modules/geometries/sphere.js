import {Vector3, computeNormal} from "../../math/math.js"
import {Geometry} from "../../core/Geometry.js";

const π = Math.PI;
const {sin, cos} = Math;


let numElements = 0;
let buffers = [];
new Geometry({numElements, buffers});

// after http://www.songho.ca/opengl/gl_sphere.html
export function createSphere(detail){

	let stackCount = detail;
	let sectorCount = detail;

	let sectorStep = 2 * π / stackCount;
	let stackStep = π / stackCount;
	
	let radius = 1;

	let n = (stackCount + 1) * (sectorCount + 1);

	let buf_position = new Float32Array(3 * n);
	let buf_color    = new Float32Array(3 * n);
	let buf_uv       = new Float32Array(2 * n);
	let buf_normal   = new Float32Array(3 * n);
	
	let counter = 0;
	for(let i = 0; i <= stackCount; i++){

		let stackAngle = π / 2 - i * stackStep;
		let xy = radius * cos(stackAngle);
		let z = radius * sin(stackAngle);

		for(let j = 0; j <= sectorCount; j++){

			let sectorAngle = j * sectorStep;

			let x = xy * cos(sectorAngle);
			let y = xy * sin(sectorAngle);

			buf_position[3 * counter + 0] = x;
			buf_position[3 * counter + 1] = y;
			buf_position[3 * counter + 2] = z;

			buf_normal[3 * counter + 0] = x;
			buf_normal[3 * counter + 1] = y;
			buf_normal[3 * counter + 2] = z;

			let u = j / sectorCount;
			let v = i / stackCount;

			buf_uv[2 * counter + 0] = u;
			buf_uv[2 * counter + 1] = v;

			buf_color[3 * counter + 0] = x;
			buf_color[3 * counter + 1] = y;
			buf_color[3 * counter + 2] = z;

			counter++;
		}
	}

	let buffers = [
		{name: "position", buffer: buf_position},
		{name: "color", buffer: buf_color},
		{name: "uv", buffer: buf_uv},
		{name: "normal", buffer: buf_normal},
	];

	let maxIndices = 2 * 3 * stackCount * sectorCount;
	let indices = new Uint32Array(maxIndices);
	let indexCounter = 0;

	for(let i = 0; i < stackCount; i++){

		let k1 = i * (sectorCount + 1);
		let k2 = k1 + sectorCount + 1;

		for(let j = 0; j < sectorCount; j++, k1++, k2++){

			if(i !== 0){
				indices[3 * indexCounter + 0] = k1;
				indices[3 * indexCounter + 1] = k2;
				indices[3 * indexCounter + 2] = k1 + 1;
				indexCounter++;
			}

			if(i !== (stackCount - 1)){
				indices[3 * indexCounter + 0] = k1 + 1;
				indices[3 * indexCounter + 1] = k2;
				indices[3 * indexCounter + 2] = k2 + 1;
				indexCounter++;
			}

		}
	}

	let clampedIndices = new Uint32Array(indices.buffer, 0, 3 * indexCounter);

	let geometry = new Geometry({
		numElements: counter,
		buffers: buffers,
		indices: clampedIndices,
	});

	return geometry;
}


export const sphere = createSphere(32);


export const cube = new Geometry({
	numElements: 36,
	buffers: [
		{
			name: "position",
			buffer: new Float32Array([
				// bottom
				-0.5, -0.5, -0.5,
				 0.5,  0.5, -0.5,
				 0.5, -0.5, -0.5,
				-0.5, -0.5, -0.5,
				-0.5,  0.5, -0.5,
				 0.5,  0.5, -0.5,

				// top
				-0.5, -0.5,  0.5,
				 0.5, -0.5,  0.5,
				 0.5,  0.5,  0.5,
				-0.5, -0.5,  0.5,
				 0.5,  0.5,  0.5,
				-0.5,  0.5,  0.5,

				// left
				-0.5, -0.5, -0.5, 
				-0.5,  0.5,  0.5, 
				-0.5,  0.5, -0.5, 
				-0.5, -0.5, -0.5, 
				-0.5, -0.5,  0.5, 
				-0.5,  0.5,  0.5, 

				// right
				 0.5, -0.5, -0.5,
				 0.5,  0.5, -0.5,
				 0.5,  0.5,  0.5,
				 0.5, -0.5, -0.5,
				 0.5,  0.5,  0.5,
				 0.5, -0.5,  0.5,

				// back
				 -0.5, 0.5, -0.5,
				  0.5, 0.5,  0.5,
				  0.5, 0.5, -0.5,
				 -0.5, 0.5, -0.5,
				 -0.5, 0.5,  0.5,
				  0.5, 0.5,  0.5,

				 // front
				 -0.5, -0.5, -0.5,
				  0.5, -0.5, -0.5,
				  0.5, -0.5,  0.5,
				 -0.5, -0.5, -0.5,
				  0.5, -0.5,  0.5,
				 -0.5, -0.5,  0.5,
			]),
		},{
			name: "color",
			buffer: new Float32Array([
				-1, -1, -1,
				 1, -1, -1,
				 1,  1, -1,
				-1, -1, -1,
				 1,  1, -1,
				-1,  1, -1,

				-1, -1,  1,
				 1, -1,  1,
				 1,  1,  1,
				-1, -1,  1,
				 1,  1,  1,
				-1,  1,  1,
			]),
		},{
			name: "uv",
			buffer: new Float32Array([
				0, 0,
				1, 0,
				1, 1,

				0, 0,
				1, 1,
				0, 1,

				0, 0,
				1, 0,
				1, 1,

				0, 0,
				1, 1,
				0, 1,
			]),
		},{
			name: "normal",
			buffer: new Float32Array([
				0, 0, -1,
				0, 0, -1,
				0, 0, -1,
				0, 0, -1,
				0, 0, -1,
				0, 0, -1,

				0, 0, 1,
				0, 0, 1,
				0, 0, 1,
				0, 0, 1,
				0, 0, 1,
				0, 0, 1,

				-1, 0, 0, 
				-1, 0, 0, 
				-1, 0, 0, 
				-1, 0, 0, 
				-1, 0, 0, 
				-1, 0, 0, 

				 1, 0, 0, 
				 1, 0, 0, 
				 1, 0, 0, 
				 1, 0, 0, 
				 1, 0, 0, 
				 1, 0, 0, 

				0, -1, 0, 
				0, -1, 0, 
				0, -1, 0, 
				0, -1, 0, 
				0, -1, 0, 
				0, -1, 0, 

				0,  1, 0, 
				0,  1, 0, 
				0,  1, 0, 
				0,  1, 0, 
				0,  1, 0, 
				0,  1, 0, 
			]),
		}
	],
});

export const cube_wireframe = new Geometry({
	numElements: 24,
	buffers: [
		{
			name: "position",
			buffer: new Float32Array([

				// bottom
				-0.5, -0.5, -0.5,
				 0.5, -0.5, -0.5,

				 0.5, -0.5, -0.5,
				 0.5,  0.5, -0.5,

				 0.5,  0.5, -0.5,
				-0.5,  0.5, -0.5,

				-0.5,  0.5, -0.5,
				-0.5, -0.5, -0.5,

				// top
				-0.5, -0.5,  0.5,
				 0.5, -0.5,  0.5,

				 0.5, -0.5,  0.5,
				 0.5,  0.5,  0.5,

				 0.5,  0.5,  0.5,
				-0.5,  0.5,  0.5,

				-0.5,  0.5,  0.5,
				-0.5, -0.5,  0.5,

				// bottom to top
				-0.5, -0.5, -0.5,
				-0.5, -0.5,  0.5,

				 0.5, -0.5, -0.5,
				 0.5, -0.5,  0.5,

				 0.5,  0.5, -0.5,
				 0.5,  0.5,  0.5,

				-0.5,  0.5, -0.5,
				-0.5,  0.5,  0.5,
			]),
		},{
			name: "direction",
			buffer: new Float32Array([

				// bottom
				 1,  0,  0,
				-1,  0,  0,

				 0,  1,  0,
				 0, -1,  0,

				-1,  0,  0,
				 1,  0,  0,

				 0, -1,  0,
				 0,  1,  0,

				// top
				 1,  0,  0,
				-1,  0,  0,

				 0,  1,  0,
				 0, -1,  0,

				-1,  0,  0,
				 1,  0,  0,

				 0, -1,  0,
				 0,  1,  0,

				// bottom to top
				0, 0,  1,
				0, 0, -1,

				0, 0,  1,
				0, 0, -1,

				0, 0,  1,
				0, 0, -1,

				0, 0,  1,
				0, 0, -1,
			]),
		}
	]
});

export const cube_wireframe_thick = new Geometry({
	numElements: 72,
	buffers: [
		{
			name: "position",
			buffer: new Float32Array([
				-0.5, -0.5, -0.5,
				+0.5, -0.5, -0.5,
				+0.5, -0.5, -0.5,
				-0.5, -0.5, -0.5,
				+0.5, -0.5, -0.5,
				-0.5, -0.5, -0.5,

				-0.5, -0.5, +0.5,
				+0.5, -0.5, +0.5,
				+0.5, -0.5, +0.5,
				-0.5, -0.5, +0.5,
				+0.5, -0.5, +0.5,
				-0.5, -0.5, +0.5,

				-0.5, +0.5, -0.5,
				+0.5, +0.5, -0.5,
				+0.5, +0.5, -0.5,
				-0.5, +0.5, -0.5,
				+0.5, +0.5, -0.5,
				-0.5, +0.5, -0.5,

				-0.5, +0.5, +0.5,
				+0.5, +0.5, +0.5,
				+0.5, +0.5, +0.5,
				-0.5, +0.5, +0.5,
				+0.5, +0.5, +0.5,
				-0.5, +0.5, +0.5,

				-0.5, -0.5, -0.5,
				-0.5, +0.5, -0.5,
				-0.5, +0.5, -0.5,
				-0.5, -0.5, -0.5,
				-0.5, +0.5, -0.5,
				-0.5, -0.5, -0.5,

				-0.5, -0.5, +0.5,
				-0.5, +0.5, +0.5,
				-0.5, +0.5, +0.5,
				-0.5, -0.5, +0.5,
				-0.5, +0.5, +0.5,
				-0.5, -0.5, +0.5,

				+0.5, -0.5, -0.5,
				+0.5, +0.5, -0.5,
				+0.5, +0.5, -0.5,
				+0.5, -0.5, -0.5,
				+0.5, +0.5, -0.5,
				+0.5, -0.5, -0.5,

				+0.5, -0.5, +0.5,
				+0.5, +0.5, +0.5,
				+0.5, +0.5, +0.5,
				+0.5, -0.5, +0.5,
				+0.5, +0.5, +0.5,
				+0.5, -0.5, +0.5,

				-0.5, -0.5, -0.5,
				-0.5, -0.5, +0.5,
				-0.5, -0.5, +0.5,
				-0.5, -0.5, -0.5,
				-0.5, -0.5, +0.5,
				-0.5, -0.5, -0.5,

				+0.5, -0.5, -0.5,
				+0.5, -0.5, +0.5,
				+0.5, -0.5, +0.5,
				+0.5, -0.5, -0.5,
				+0.5, -0.5, +0.5,
				+0.5, -0.5, -0.5,

				+0.5, +0.5, -0.5,
				+0.5, +0.5, +0.5,
				+0.5, +0.5, +0.5,
				+0.5, +0.5, -0.5,
				+0.5, +0.5, +0.5,
				+0.5, +0.5, -0.5,

				-0.5, +0.5, -0.5,
				-0.5, +0.5, +0.5,
				-0.5, +0.5, +0.5,
				-0.5, +0.5, -0.5,
				-0.5, +0.5, +0.5,
				-0.5, +0.5, -0.5,
			]),
		},{
			name: "direction",
			buffer: new Float32Array([
				+1, 0, 0,
				+1, 0, 0,
				-1, 0, 0,
				+1, 0, 0,
				-1, 0, 0,
				-1, 0, 0,

				+1, 0, 0,
				+1, 0, 0,
				-1, 0, 0,
				+1, 0, 0,
				-1, 0, 0,
				-1, 0, 0,

				+1, 0, 0,
				+1, 0, 0,
				-1, 0, 0,
				+1, 0, 0,
				-1, 0, 0,
				-1, 0, 0,

				+1, 0, 0,
				+1, 0, 0,
				-1, 0, 0,
				+1, 0, 0,
				-1, 0, 0,
				-1, 0, 0,

				0, +1, 0,
				0, +1, 0,
				0, -1, 0,
				0, +1, 0,
				0, -1, 0,
				0, -1, 0,

				0, +1, 0,
				0, +1, 0,
				0, -1, 0,
				0, +1, 0,
				0, -1, 0,
				0, -1, 0,

				0, +1, 0,
				0, +1, 0,
				0, -1, 0,
				0, +1, 0,
				0, -1, 0,
				0, -1, 0,

				0, +1, 0,
				0, +1, 0,
				0, -1, 0,
				0, +1, 0,
				0, -1, 0,
				0, -1, 0,

				0, 0, +1,
				0, 0, +1,
				0, 0, -1,
				0, 0, +1,
				0, 0, -1,
				0, 0, -1,

				0, 0, +1,
				0, 0, +1,
				0, 0, -1,
				0, 0, +1,
				0, 0, -1,
				0, 0, -1,

				0, 0, +1,
				0, 0, +1,
				0, 0, -1,
				0, 0, +1,
				0, 0, -1,
				0, 0, -1,

				0, 0, +1,
				0, 0, +1,
				0, 0, -1,
				0, 0, +1,
				0, 0, -1,
				0, 0, -1,
			]),
		}
	]
});
