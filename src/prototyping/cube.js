
export const cube = {
	vertexCount: 12,
	triangleCount: 4,
	buffers: [
		{
			name: "position",
			buffer: new Float32Array([
				// bottom
				-1, -1, -1,
				1, -1, -1,
				1, -1,  1,

				-1, -1, -1,
				1, -1,  1,
				-1, -1,  1,

				// top
				-1,  1, -1,
				1,  1, -1,
				1,  1,  1,

				-1,  1, -1,
				1,  1,  1,
				-1,  1,  1,
			]),
		},{
			name: "color",
			buffer: new Float32Array([
				1, 0, 0, 1,
				0, 1, 0, 1,
				0, 0, 1, 1,
	
				1, 0, 0, 1,
				0, 1, 0, 1,
				0, 0, 1, 1,

				1, 0, 0, 1,
				0, 1, 0, 1,
				0, 0, 1, 1,
	
				1, 0, 0, 1,
				0, 1, 0, 1,
				0, 0, 1, 1,
			]),
		}
	]

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
