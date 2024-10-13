
import {Vector3, Mesh, Geometry, WireframeMaterial, TriangleColorMaterial} from "potree";

function randomPermutation(n, max){

	let i64 = new BigInt64Array(max);
	let i32 = new Int32Array(i64.buffer);

	for(let i = 0; i < max; i++){
		// most significant bits for shuffle value
		i32[2 * i + 1] = Math.random() * 100_000_000;
		// least significant bits for index value
		i32[2 * i + 0] = i;
	}

	i64.sort();

	let result = new Uint32Array(n);
	for(let i = 0; i < n; i++){
		result[i] = i32[2 * i + 0];
	}

	return result;
}


export function generateLOD(node){

	let geometry = node.geometry;
	let positions = new Float32Array(node.geometry.buffers[2].buffer.buffer);
	let indices = geometry.indices;
	let numVertices = positions.length / 3;
	let numTriangles = indices.length / 3;
	let triangleIDs = new Uint32Array(numTriangles);

	let indexIndices = new Uint32Array(indices.length);
	for(let i = 0; i < indexIndices.length; i++){
		indexIndices[i] = i;
	}

	for(let i = 0; i < triangleIDs.length; i++){
		triangleIDs[i] = i;
	}


	let edges = [];
	let tmp0 = new Vector3();
	let tmp1 = new Vector3();

	let lengthBetween = (i0, i1) => {
		tmp0.set(positions[3 * i0 + 0], positions[3 * i0 + 1], positions[3 * i0 + 2]);
		tmp1.set(positions[3 * i1 + 0], positions[3 * i1 + 1], positions[3 * i1 + 2]);

		return tmp0.distanceTo(tmp1);
	};

	let indexToPosition = (index) => {

		let realIndex = indexIndices[index];

		return new Vector3(
			positions[3 * realIndex + 0], 
			positions[3 * realIndex + 1], 
			positions[3 * realIndex + 2]
		);
	};

	for(let triangleID = 0; triangleID < numTriangles; triangleID++){
		let i0 = indices[3 * triangleID + 0];
		let i1 = indices[3 * triangleID + 1];
		let i2 = indices[3 * triangleID + 2];

		let e01 = {length: lengthBetween(i0, i1), start: i0, end: i1};
		let e12 = {length: lengthBetween(i1, i2), start: i1, end: i2};
		let e20 = {length: lengthBetween(i2, i0), start: i2, end: i0};

		edges.push(e01, e12, e20);
	}

	let collapsedEdges = [];
	let threshold = 0.015;
	for(let edgeID = 0; edgeID < edges.length; edgeID++){

		let edge = edges[edgeID];

		if(edge.length < threshold){
			// collapse
			// update adjacent
			// add collapsed to visualization
			let start = indexToPosition(edge.start);
			let end = indexToPosition(edge.end);


			// collapse
			let center = start.clone().add(end).multiplyScalar(0.5);
			let realStartIndex = indexIndices[edge.start];
			let realEndIndex = indexIndices[edge.end];
			positions[3 * realStartIndex + 0] = center.x;
			positions[3 * realStartIndex + 1] = center.y;
			positions[3 * realStartIndex + 2] = center.z;

			positions[3 * realEndIndex + 0] = center.x;
			positions[3 * realEndIndex + 1] = center.y;
			positions[3 * realEndIndex + 2] = center.z;

			indexIndices[edge.end] = indexIndices[edge.start];

			start.applyMatrix4(node.world);
			end.applyMatrix4(node.world);
			collapsedEdges.push([start, end]);
		}

	}


	// {
	// 	let geometry = new Geometry();
	// 	geometry.buffers = [
	// 		...node.geometry.buffers,
	// 		{name: "triangle_ids", buffer: new Uint8Array(triangleIDs.buffer)},
	// 	];


	// 	geometry.indices = node.geometry.indices;

	// 	let node2 = new Mesh("wireframe", geometry);

	// 	node2.material = new TriangleColorMaterial();
	// 	node2.material.color.set(0, 1, 0);
	// 	node2.position.copy(node.position);
	// 	node2.scale.copy(node.scale);
	// 	node2.rotation.copy(node.rotation);

	// 	scene.root.children.push(node2);
	// }


	{
		
		let query = new Vector3(-15.1, -6.3, 1.4);
		let pool = new Float32Array(scene.root.children[2].geometry.buffers[2].buffer.buffer);
		let numPoints = pool.length / 3;
		let results = [];

		for(let i = 0; i < numPoints; i++){

			let point = new Vector3(
				pool[3 * i + 0],
				pool[3 * i + 1],
				pool[3 * i + 2],
			);
			point.applyMatrix4(node.world);

			let distance = point.distanceTo(query);

			if(distance < 0.062){
				console.log("vertexID: ", i);
				// console.log(distance);
				// console.log(point);

				// break;
			}


		}

	}


	potree.onUpdate( () => {

		let color = new Vector3(255, 0, 0);
		// for(let edge of collapsedEdges){
		// 	potree.renderer.drawLine(edge[0], edge[1], color);
		// }

		// for(let point of centers){
			// potree.renderer.drawBox(
			// 	point, 
			// 	new Vector3(
			// 		0.01, 
			// 		0.01, 
			// 		0.01),
			// 	new Vector3(255, 255, 0),
			// );
		// }
	});



}