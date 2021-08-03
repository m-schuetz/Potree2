// function simplify(node){

	// 	let geometry = node.geometry;
	// 	let positions = new Float32Array(node.geometry.buffers[2].buffer.buffer);
	// 	let indices = geometry.indices;
	// 	let numVertices = positions.length / 3;
	// 	let numTriangles = indices.length / 3;

	// 	let max = new Vector3(-Infinity, -Infinity, -Infinity);
	// 	let tmp = new Vector3();

	// 	node.updateWorld();

	// 	let edges = [];
	// 	window.edges = edges;

	// 	let p0 = new Vector3();
	// 	let p1 = new Vector3();
	// 	let p2 = new Vector3();

	// 	for(let i = 0; i < numTriangles; i++){
	// 		let i0 = indices[3 * i + 0];
	// 		let i1 = indices[3 * i + 1];
	// 		let i2 = indices[3 * i + 2];

	// 		p0.set(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
	// 		p1.set(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);
	// 		p2.set(positions[3 * i2 + 0], positions[3 * i2 + 1], positions[3 * i2 + 2]);

	// 		let d01 = p0.distanceTo(p1);
	// 		let d12 = p1.distanceTo(p2);
	// 		let d20 = p2.distanceTo(p0);

	// 		let edge_01 = {
	// 			index_0: i0, 
	// 			index_1: i1,
	// 			length: d01,
	// 		};

	// 		edges.push(edge_01);
	// 	}

	// 	edges.sort((a, b) => {
	// 		return a.length - b.length;
	// 	});

	// 	let lines = [];

	// 	for(let i = 0; i < 50_000; i++){
	// 		let edge = edges[i];

	// 		let i0 = edge.index_0;
	// 		let i1 = edge.index_1;

	// 		let start = new Vector3(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
	// 		let end   = new Vector3(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);

	// 		start.applyMatrix4(node.world);
	// 		end.applyMatrix4(node.world);

	// 		lines.push([start, end]);
	// 	}

	// 	potree.onUpdate( () => {

	// 		for(let line of lines){

	// 			potree.renderer.drawLine(
	// 				line[0], line[1],
	// 				new Vector3(255, 0, 0),
	// 			);

	// 		}
	// 	});
	// }

	// function simplify(node){

	// 	let geometry = node.geometry;
	// 	let positions = new Float32Array(node.geometry.buffers[2].buffer.buffer);
	// 	let indices = geometry.indices;
	// 	let numVertices = positions.length / 3;
	// 	let numTriangles = indices.length / 3;

	// 	let tmpPositions = new Float32Array(positions);
	// 	let indexIndices = new Uint32Array(indices);

	// 	let max = new Vector3(-Infinity, -Infinity, -Infinity);
	// 	let tmp = new Vector3();

	// 	node.updateWorld();

	// 	let edges = [];
	// 	window.edges = edges;

	// 	let p0 = new Vector3();
	// 	let p1 = new Vector3();
	// 	let p2 = new Vector3();

	// 	for(let i = 0; i < numTriangles; i++){
	// 		let i0 = indices[3 * i + 0];
	// 		let i1 = indices[3 * i + 1];
	// 		let i2 = indices[3 * i + 2];

	// 		p0.set(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
	// 		p1.set(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);
	// 		p2.set(positions[3 * i2 + 0], positions[3 * i2 + 1], positions[3 * i2 + 2]);

	// 		let d01 = p0.distanceTo(p1);
	// 		let d12 = p1.distanceTo(p2);
	// 		let d20 = p2.distanceTo(p0);

	// 		let edge_01 = {
	// 			index_0: i0, 
	// 			index_1: i1,
	// 			length: d01,
	// 		};

	// 		edges.push(edge_01);
	// 	}

	// 	edges.sort((a, b) => {
	// 		return a.length - b.length;
	// 	});

	// 	let lines = [];
	// 	let boxes = [];

	// 	for(let i = 0; i < 150_000; i++){
	// 		let edge = edges[i];

	// 		let i0 = edge.index_0;
	// 		let i1 = edge.index_1;

	// 		let start = new Vector3(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
	// 		let end   = new Vector3(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);

	// 		// COLLAPSE!

	// 		let collapsedVertex = start.clone().add(end).multiplyScalar(0.5);
	// 		tmpPositions[3 * i0 + 0] = collapsedVertex.x;
	// 		tmpPositions[3 * i0 + 1] = collapsedVertex.y;
	// 		tmpPositions[3 * i0 + 2] = collapsedVertex.z;
	// 		tmpPositions[3 * i1 + 0] = collapsedVertex.x;
	// 		tmpPositions[3 * i1 + 1] = collapsedVertex.y;
	// 		tmpPositions[3 * i1 + 2] = collapsedVertex.z;


	// 		start.applyMatrix4(node.world);
	// 		end.applyMatrix4(node.world);

	// 		lines.push([start, end]);

	// 		collapsedVertex.applyMatrix4(node.world);
	// 		boxes.push([
	// 			collapsedVertex,
	// 			new Vector3(1, 1, 1).multiplyScalar(0.1 * start.distanceTo(end)),
	// 		]);
	// 	}

	// 	// potree.onUpdate( () => {

	// 	// 	for(let line of lines){
	// 	// 		potree.renderer.drawLine(
	// 	// 			line[0], line[1],
	// 	// 			new Vector3(255, 0, 0),
	// 	// 		);
	// 	// 	}

	// 	// 	for(let box of boxes){
	// 	// 		potree.renderer.drawBox(
	// 	// 			box[0], box[1],
	// 	// 			new Vector3(0, 0, 255),
	// 	// 		);
	// 	// 	}
	// 	// });



	// 	{ // add wireframe
	// 		let geometry = new Geometry();
	// 		geometry.buffers = [
	// 			node.geometry.buffers[0],
	// 			node.geometry.buffers[1],
	// 			{name: "position", buffer: new Uint8Array(tmpPositions.buffer)},
	// 			node.geometry.buffers[3],
	// 		];
	// 		geometry.indices = node.geometry.indices;

	// 		let simplified = new Mesh("simplified", geometry);

	// 		simplified.material = new WireframeMaterial();
	// 		simplified.material.color.set(1, 0, 0);
	// 		simplified.position.copy(node.position);
	// 		simplified.scale.copy(node.scale);
	// 		simplified.rotation.copy(node.rotation);

	// 		scene.root.children.push(simplified);
	// 	}




	// }