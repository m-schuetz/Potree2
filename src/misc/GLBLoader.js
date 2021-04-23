
import {Geometry} from "../core/Geometry.js";
import {SceneNode} from "../scene/SceneNode.js";
import {Mesh} from "../modules/mesh/Mesh.js";
import {PhongMaterial, ColorMode} from "../modules/mesh/PhongMaterial.js";
import {Box3, Vector3} from "../math/math.js";



export async function load(url){


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

	console.log(json);

	let binaryChunkOffset = 12 + 8 + chunkSize + 8;
	let binaryChunkSize = view.getUint32(20 + chunkSize, true);
	
	let bufferViews = [];
	for(let glbBufferView of json.bufferViews){

		let offset = glbBufferView.byteOffset;
		let size = glbBufferView.byteLength;

		let data = buffer.slice(binaryChunkOffset + offset, binaryChunkOffset + offset + size);

		bufferViews.push({buffer: data});
	}


	let geometry = new Geometry();

	let glbMesh = json.meshes[0];
	let glbPrimitive = glbMesh.primitives[0];
	for(let attributeName of Object.keys(glbPrimitive.attributes)){
		let accessorRef = glbPrimitive.attributes[attributeName];
		let accessor = json.accessors[accessorRef];
		let bufferView = bufferViews[accessor.bufferView];

		let mappedName = {
			"COLOR_0": "color",
			"NORMAL": "normal",
			"POSITION": "position",
			"TEXCOORD_0": "uv",
		}[attributeName] ?? attributeName;

		let geomBuffer = {name: mappedName, buffer: new Uint8Array(bufferView.buffer)};
		geometry.buffers.push(geomBuffer);

		if(glbPrimitive.indices){
			let accessor = json.accessors[glbPrimitive.indices];
			let bufferView = bufferViews[accessor.bufferView];

			geometry.indices = new Uint32Array(bufferView.buffer);
		}
	}

	console.log(geometry);

	
	// BOUNDING BOX
	let boundingBox = new Box3(); {

		let buffer = geometry.buffers.find(b => b.name === "position").buffer;
		let f32 = new Float32Array(buffer.buffer);

		let tmp = new Vector3();
		for(let i = 0; i < f32.length / 3; i++){
			tmp.x = f32[3 * i + 0];
			tmp.y = f32[3 * i + 1];
			tmp.z = f32[3 * i + 2];
			boundingBox.expandByPoint(tmp);
		}
	}

	geometry.boundingBox = boundingBox;

	let mesh = new Mesh("glb mesh", geometry);
	let node = new SceneNode("glb node");
	node.children.push(mesh);

	{ // IMAGES
		let image = json.images[0];
		let buffer = bufferViews[image.bufferView].buffer;
		let mimeType = image.mimeType;

		var u8 = new Uint8Array(buffer);
		var blob = new Blob([u8], {type: mimeType});
		var imageUrl = URL.createObjectURL(blob);
		var img = document.createElement("img");
		
		img.src = imageUrl;
		await img.decode();

		let imageBitmap = await createImageBitmap(img);

		mesh.material = new PhongMaterial();
		mesh.material.image = imageBitmap;
		mesh.material.colorMode = ColorMode.TEXTURE;
	}

	let duration = performance.now() - tStart;
	console.log("duration: " + duration  + "ms");


	

	return node;

};



