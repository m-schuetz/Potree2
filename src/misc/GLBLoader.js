
import {Geometry, SceneNode, Mesh} from "potree";
import {Box3, Vector3} from "potree";
import {PhongMaterial, ColorMode} from "potree";
import {NormalMaterial, WireframeMaterial} from "potree";

let tmpCanvas = null;
let tmpContext = null;
function getTmpContext(){

	if(tmpCanvas === null){
		tmpCanvas = document.createElement('canvas');
		tmpCanvas.width = 8192;
		tmpCanvas.height = 8192;
		tmpContext = tmpCanvas.getContext('2d');
	}

	return tmpContext;
}

export function load(url, callbacks){

	let workerPath = "./src/misc/GLBLoaderWorker.js";
	let worker = new Worker(workerPath, {type: "module"});

	let root = new SceneNode("glb root");

	let images = new Map();

	let image_loaded = (e) => {
		images.set(e.data.imageRef, e.data.imageBitmap);
	};

	let mesh_batch_loaded = (e) => {

		let imageBitmap = null;
		if(images.has(e.data.imageRef)){
			imageBitmap = images.get(e.data.imageRef);
		}

		let geometryData = e.data.geometry;

		let geometry = new Geometry();
		geometry.buffers = geometryData.buffers;
		geometry.indices = geometryData.indices;
		geometry.numElements = geometryData.numElements;
		geometry.boundingBox.min.copy(geometryData.boundingBox.min);
		geometry.boundingBox.max.copy(geometryData.boundingBox.max);

		let mesh = new Mesh("glb mesh", geometry);

		if(imageBitmap){
			mesh.material = new PhongMaterial();
			mesh.material.image = imageBitmap;
			mesh.material.colorMode = ColorMode.TEXTURE;
			mesh.material.imageBuffer = e.data.imageBuffer;
		}else{
			mesh.material = new PhongMaterial();
			mesh.material.image = null;
			mesh.material.colorMode = ColorMode.VERTEX_COLOR;
		}

		root.children.push(mesh);

		if(root.children.length === 1){
			callbacks.onStart(root);
			callbacks.onNode(mesh);
		}else{
			callbacks.onNode(mesh);
		}
	};

	

	worker.onmessage = (e) => {

		if(e.data.type === "mesh_batch_loaded"){
			mesh_batch_loaded(e);
		}else if(e.data.type === "image_loaded"){
			image_loaded(e);
		}

	};

	let absoluteUrl = new URL(url, document.baseURI).href;
	worker.postMessage({url: absoluteUrl});


};



