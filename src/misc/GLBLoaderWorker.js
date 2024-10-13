
import {Vector3, Box3} from "potree";

function makeGeometry(numVertices){
	let positions = new Float32Array(3 * numVertices);
	let uvs = new Float32Array(2 * numVertices);

	let geometry = {
		buffers: [
			{name: "position", buffer: positions},
			{name: "uv", buffer: uvs},
		],
		numElements: numVertices,
	};

	return geometry;
}

let imageMap = new Map();

async function loadImage(imageRef, json, bufferViews, onImageLoaded){

	let image = json.images[imageRef];

	if(imageMap.has(image.bufferView)){
		return null;
	}else{

		let buffer = bufferViews[image.bufferView].buffer;
		let mimeType = image.mimeType;

		var u8 = new Uint8Array(buffer);
		var blob = new Blob([u8], {type: mimeType});

		let imageBitmap = await createImageBitmap(blob);

		let message = {
			type: "image_loaded",
			imageBitmap: imageBitmap,
			imageRef: imageRef
		};

		let transferables = [];

		postMessage(message, transferables);

		imageMap.set(image.bufferView, "loaded");

		return imageBitmap;
	}
}

async function loadMesh(mesh, json, bufferViews, onBatchLoaded){

	let numIndices = 0;
	for(let primitive of mesh.primitives){
		if(typeof primitive.indices !== "undefined"){
			let indices_accessor = json.accessors[primitive.indices];

			numIndices += indices_accessor.count;
		}else{
			// skip for now
		}
	}

	let maxTrianglesPerBatch = 500_000;

	let batches = [];
	let images = [];

	for(let primitive of mesh.primitives){

		if(typeof primitive.indices === "undefined"){
			continue;
		}

		let accessor = json.accessors[primitive.indices];
		let bufferView = bufferViews[accessor.bufferView];

		let num_indices = 0;
		if(accessor.componentType == 5123){
			num_indices = bufferView.buffer.byteLength / 2;
		}else{
			num_indices = bufferView.buffer.byteLength / 4;
		}

		let numTriangles_primitive = num_indices / 3;
		let numBatches = Math.ceil(numTriangles_primitive / maxTrianglesPerBatch);
		for(let i = 0; i < numBatches; i++){

			let firstTriangle = i * maxTrianglesPerBatch;
			let numTriangles = Math.min(numTriangles_primitive - firstTriangle, maxTrianglesPerBatch);

			let batch = {primitive, firstTriangle, numTriangles};

			batches.push(batch);
		}

		let materialRef = primitive.material;
		let material = json.materials[materialRef];
		if(material.pbrMetallicRoughness.baseColorTexture){
			let imageRef = material.pbrMetallicRoughness.baseColorTexture.index;
		}
	}

	for(let batch of batches){

		let {primitive, firstTriangle, numTriangles} = batch;

		let imageRef = null;
		{ // load image, if available
			let materialRef = mesh.primitives[0].material;
			let material = json.materials[materialRef];
			imageRef = material.pbrMetallicRoughness.baseColorTexture.index;

			await loadImage(imageRef, json, bufferViews);
		}

		let boundingBox = new Box3();

		let positions = new Float32Array(9 * numTriangles);
		let uvs = new Float32Array(6 * numTriangles);

		let prim_indices;
		{
			let accessor = json.accessors[primitive.indices];
			let bufferView = bufferViews[accessor.bufferView];

			if(accessor.componentType == 5123){
				prim_indices = new Uint32Array(new Uint16Array(bufferView.buffer))
			}else{
				prim_indices = new Uint32Array(bufferView.buffer);
			}
		}

		if(typeof primitive.attributes.POSITION !== "undefined"){
			let accessorRef = primitive.attributes.POSITION ?? null;
			let accessor = json.accessors[accessorRef];
			let bufferView = bufferViews[accessor.bufferView];
			let prim_positions = new Float32Array(bufferView.buffer)

			for(let i = 0; i < numTriangles; i++){
				let i0 = prim_indices[3 * (firstTriangle + i) + 0];
				let i1 = prim_indices[3 * (firstTriangle + i) + 1];
				let i2 = prim_indices[3 * (firstTriangle + i) + 2];

				positions[9 * i + 0] = prim_positions[3 * i0 + 0];
				positions[9 * i + 1] = prim_positions[3 * i0 + 1];
				positions[9 * i + 2] = prim_positions[3 * i0 + 2];
				positions[9 * i + 3] = prim_positions[3 * i1 + 0];
				positions[9 * i + 4] = prim_positions[3 * i1 + 1];
				positions[9 * i + 5] = prim_positions[3 * i1 + 2];
				positions[9 * i + 6] = prim_positions[3 * i2 + 0];
				positions[9 * i + 7] = prim_positions[3 * i2 + 1];
				positions[9 * i + 8] = prim_positions[3 * i2 + 2];

				boundingBox.expandByXYZ(
					prim_positions[3 * i0 + 0],
					prim_positions[3 * i0 + 1],
					prim_positions[3 * i0 + 2],
				);
				boundingBox.expandByXYZ(
					prim_positions[3 * i1 + 0],
					prim_positions[3 * i1 + 1],
					prim_positions[3 * i1 + 2],
				);
				boundingBox.expandByXYZ(
					prim_positions[3 * i2 + 0],
					prim_positions[3 * i2 + 1],
					prim_positions[3 * i2 + 2],
				);
			}
		}else{
			throw "wat?";
		}

		if(typeof primitive.attributes.TEXCOORD_0 !== "undefined"){
			let accessorRef = primitive.attributes.TEXCOORD_0 ?? null;
			let accessor = json.accessors[accessorRef];
			let bufferView = bufferViews[accessor.bufferView];
			let prim_uvs = new Float32Array(bufferView.buffer);

			for(let i = 0; i < numTriangles; i++){
				let i0 = prim_indices[3 * (firstTriangle + i) + 0];
				let i1 = prim_indices[3 * (firstTriangle + i) + 1];
				let i2 = prim_indices[3 * (firstTriangle + i) + 2];

				uvs[6 * i + 0] = prim_uvs[2 * i0 + 0];
				uvs[6 * i + 1] = prim_uvs[2 * i0 + 1];
				uvs[6 * i + 2] = prim_uvs[2 * i1 + 0];
				uvs[6 * i + 3] = prim_uvs[2 * i1 + 1];
				uvs[6 * i + 4] = prim_uvs[2 * i2 + 0];
				uvs[6 * i + 5] = prim_uvs[2 * i2 + 1];
			}
		}else{
			// no uvs
		}

		let geometry = {
			buffers: [
				{name: "position", buffer: positions},
				{name: "uv", buffer: uvs},
			],
			numElements: 3 * numTriangles,
			boundingBox: boundingBox,
		};

		onBatchLoaded({geometry, imageRef});
	}

}

onmessage = async function(e){

	let {url} = e.data;

	let response = await fetch(url);
	let buffer = await response.arrayBuffer();
	let view = new DataView(buffer);

	let tStart = performance.now();

	let magic = view.getUint32(0, true);
	let version = view.getUint32(4, true);
	let length = view.getUint32(8, true);
	let chunkSize = view.getUint32(12, true);
	let chunkType = view.getUint32(16, true);

	let jsonBuffer = buffer.slice(20, 20 + chunkSize);
	let decoder = new TextDecoder();
	let jsonText = decoder.decode(jsonBuffer);
	let json = JSON.parse(jsonText);

	let binaryChunkOffset = 12 + 8 + chunkSize + 8;
	let binaryChunkSize = view.getUint32(20 + chunkSize, true);
	
	let bufferViews = [];
	for(let glbBufferView of json.bufferViews){

		let offset = glbBufferView.byteOffset;
		let size = glbBufferView.byteLength;

		let data = buffer.slice(binaryChunkOffset + offset, binaryChunkOffset + offset + size);

		bufferViews.push({buffer: data});
	}

	let promises = [];
	
	for(let glbMesh of json.meshes){

		let promise = loadMesh(glbMesh, json, bufferViews, (result) => {

			let message = {
				type: "mesh_batch_loaded",
				geometry: result.geometry,
				imageRef: result.imageRef,
			};

			let transferables = [
				...result.geometry.buffers.map(b => b.buffer.buffer),
			];

			let duration = performance.now() - tStart;
			console.log("duration: " + duration  + "ms");

			postMessage(message, transferables);
		});

		promises.push(promise);
	}

	{
		await Promise.all(promises);

		let message = {
			type: "finished",
		};
		
		postMessage(message);
	}
};