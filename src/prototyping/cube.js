import {Vector3, computeNormal} from "potree";

export const cube = {
	vertexCount: 36,
	triangleCount: 4,
	buffers: [
		{
			name: "position",
			buffer: new Float32Array([
				// bottom
				-1, -1, -1,
				 1, -1, -1,
				 1,  1, -1,
				-1, -1, -1,
				 1,  1, -1,
				-1,  1, -1,

				// top
				-1, -1,  1,
				 1, -1,  1,
				 1,  1,  1,
				-1, -1,  1,
				 1,  1,  1,
				-1,  1,  1,

				// left
				-1, -1, -1, 
				-1,  1, -1, 
				-1,  1,  1, 
				-1, -1, -1, 
				-1,  1,  1, 
				-1, -1,  1, 

				// right
				 1, -1, -1,
				 1,  1, -1,
				 1,  1,  1,
				 1, -1, -1,
				 1,  1,  1,
				 1, -1,  1,

				// back
				 -1, 1, -1,
				  1, 1, -1,
				  1, 1,  1,
				 -1, 1, -1,
				  1, 1,  1,
				 -1, 1,  1,

				 // front
				 -1, -1, -1,
				  1, -1, -1,
				  1, -1,  1,
				 -1, -1, -1,
				  1, -1,  1,
				 -1, -1,  1,
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
};


export const cube_indexed = {
	vertexCount: 12,
	triangleCount: 4,
	buffers: [
		{
			name: "position",
			buffer: new Float32Array([
				-1, -1, -1,
				 1, -1, -1,
				 1,  1, -1,
				-1,  1, -1,

				-1, -1,  1,
				 1, -1,  1,
				 1,  1,  1,
				-1,  1,  1,
			]),
		},{
			name: "color",
			buffer: new Float32Array([
				 0,  0,  0,
				 1,  0,  0,
				 1,  1,  0,
				 0,  1,  0,

				 0,  0,  1,
				 1,  0,  1,
				 1,  1,  1,
				 0,  1,  1,
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
		}
	],
	indices: new Uint32Array([
		0, 1, 2,
		0, 2, 3,

		4, 5, 6,
		4, 6, 7,

		0, 1, 5,
		0, 5, 4,

		1, 2, 6,
		1, 6, 5,

		2, 3, 7, 
		2, 7, 6,

		3, 0, 4,
		3, 4, 7,
	]),

};


function createPointCube(n){

	let numPoints = (n * n * n);
	let position = new Float32Array(3 * numPoints);
	let color = new Float32Array(4 * numPoints);

	let processed = 0;
	for(let i = 0; i < n; i++){
	for(let j = 0; j < n; j++){
	for(let k = 0; k < n; k++){

		position[3 * processed + 0] = i / n + Math.random() / n;
		position[3 * processed + 1] = j / n + Math.random() / n;
		position[3 * processed + 2] = k / n + Math.random() / n;

		color[4 * processed + 0] = i / n;
		color[4 * processed + 1] = j / n;
		color[4 * processed + 2] = k / n;
		color[4 * processed + 3] = 1;

		processed++;
	}
	}
	}

	let object = {
		vertexCount: numPoints,
		buffers: [{
				name: "position",
				buffer: position,
			},{
				name: "color",
				buffer: color,
			}
		]
	};

	return object;
}

export let pointCube = createPointCube(10);


export function createWave(){

	let n = 500;

	let sample = (u, v) => {

		let x = u;
		let y = v;
		let z = Math.sin(u) * Math.cos(v);

		return new Vector3(x, y, z);
	};

	let numTriangles = n * n * 2;
	let numVertices = numTriangles * 3;
	
	let positions = new Float32Array(3 * numVertices);
	let colors = new Float32Array(4 * numVertices);
	let uvs = new Float32Array(2 * numVertices);
	let normals = new Float32Array(3 * numVertices);

	let k = 0;
	let addVertex = (pos, col, uv, normal) => {

		positions[3 * k + 0] = pos.x;
		positions[3 * k + 1] = pos.y;
		positions[3 * k + 2] = pos.z;

		colors[4 * k + 0] = col[0];
		colors[4 * k + 1] = col[1];
		colors[4 * k + 2] = col[2];
		colors[4 * k + 3] = col[3];

		uvs[2 * k + 0] = uv[0];
		uvs[2 * k + 1] = uv[1];

		normals[3 * k + 0] = normal.x;
		normals[3 * k + 1] = normal.y;
		normals[3 * k + 2] = normal.z;

		k++;
	};

	for(let i = 0; i < n; i++){
		for(let j = 0; j < n; j++){

			let u0 = 10 * Math.PI * ((i + 0) / n) - 5 * Math.PI;
			let u1 = 10 * Math.PI * ((i + 1) / n) - 5 * Math.PI;
			let v0 = 10 * Math.PI * ((j + 0) / n) - 5 * Math.PI;
			let v1 = 10 * Math.PI * ((j + 1) / n) - 5 * Math.PI;

			let p0 = sample(u0, v0);
			let p1 = sample(u1, v0);
			let p2 = sample(u1, v1);
			let p3 = sample(u0, v1);

			let n1 = computeNormal(p0, p1, p2);
			let n2 = computeNormal(p0, p2, p3);

			addVertex(p0, [0.2 + 0.8 * p0.z, 0.2, 0.2, 0.0], [u0, v0], n1);
			addVertex(p1, [0.2 + 0.8 * p1.z, 0.2, 0.2, 0.0], [u1, v0], n1);
			addVertex(p2, [0.2 + 0.8 * p2.z, 0.2, 0.2, 0.0], [u1, v1], n1);
			addVertex(p0, [0.2 + 0.8 * p0.z, 0.2, 0.2, 0.0], [u0, v0], n2);
			addVertex(p2, [0.2 + 0.8 * p2.z, 0.2, 0.2, 0.0], [u1, v1], n2);
			addVertex(p3, [0.2 + 0.8 * p3.z, 0.2, 0.2, 0.0], [u0, v1], n2);
		}
	}

	let geometry = {
		vertexCount: numVertices,
		triangleCount: numTriangles,
		buffers: [
			{name: "position", buffer: positions},
			{name: "color", buffer: colors},
			{name: "uv", buffer: uvs},
			{name: "normal", buffer: normals},
		],
	};

	return geometry;

}