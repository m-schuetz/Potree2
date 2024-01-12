
import { Vector3 } from "../../Potree.js";
import {TDTiles, TDTilesNode, BVSphere} from "./3DTiles.js";

let dbgCount = 0;
const SPECTRAL = [
	[158,1,66],
	[213,62,79],
	[244,109,67],
	[253,174,97],
	[254,224,139],
	[255,255,191],
	[230,245,152],
	[171,221,164],
	[102,194,165],
	[50,136,189],
	[94,79,162],
];

export class TDTilesLoader{

	constructor(){
		this.tiles = null;
	}

	async parseB3dm(buffer){
		let view = new DataView(buffer);

		let version                      = view.getUint32(4, true);
		let byteLength                   = view.getUint32(8, true);
		let featureTableJsonByteLength   = view.getUint32(12, true);
		let featureTableBinaryByteLength = view.getUint32(16, true);
		let batchTableJSONByteLength     = view.getUint32(20, true);
		let batchTableBinaryByteLength   = view.getUint32(24, true);

		let featureStart = 28;
		let batchStart = featureStart + featureTableJsonByteLength + featureTableBinaryByteLength;
		let gltfStart = batchStart + batchTableJSONByteLength + batchTableBinaryByteLength;

		let batchu8 = new Uint8Array(buffer, featureStart, featureTableJsonByteLength);

		let dec = new TextDecoder("utf-8");

		let featureJsonStr = dec.decode(batchu8);
		let featureJson    = JSON.parse(featureJsonStr);

		let gltf = {};
		{

			let magic            = view.getUint32(gltfStart + 0, true);
			let version          = view.getUint32(gltfStart + 4, true);
			let length           = view.getUint32(gltfStart + 8, true);

			let jsStart = gltfStart + 12;
			let chunk_js_length  = view.getUint32(jsStart + 0, true);
			let chunk_js_type    = view.getUint32(jsStart + 4, true);
			let chunk_js_data    = new Uint8Array(buffer, jsStart + 8, chunk_js_length);
			
			let strJson = dec.decode(chunk_js_data);
			let json = JSON.parse(strJson);

			let binStart           = gltfStart + 20 + chunk_js_length;
			let chunk_bin_length   = view.getUint32(binStart + 0, true);
			let chunk_bin_type     = view.getUint32(binStart + 4, true);
			let chunk_bin_data     = new Uint8Array(buffer, binStart + 8, chunk_bin_length);

			let imageBufferViewRef = json.images[0].bufferView;
			let imageBufferView    = json.bufferViews[imageBufferViewRef];

			let jpegU8 = new Uint8Array(
				buffer, 
				binStart + 8 + imageBufferView.byteOffset, 
				imageBufferView.byteLength
			);

			// LOAD JPEG ENCODED COLORS
			{ // try loading image
				let blob = new Blob([jpegU8]);

				let image = await createImageBitmap(blob);
				gltf.image = image;
			}
			
			gltf.json = json;
			gltf.buffer = new Uint8Array(buffer, gltfStart, length);
			gltf.chunks = [
				{type: "JSON", start: jsStart, length: chunk_js_length},
				{type: "BIN", start: binStart, length: chunk_bin_length},
			];

		}
		
		let b3dm = {
			buffer,
			json: featureJson,
			version, byteLength,
			gltf,
		};

		// console.log({
		// 	version,
		// 	byteLength,
		// 	featureTableJsonByteLenght,
		// 	featureTableBinaryByteLength,
		// 	batchTableJSONByteLength,
		// 	batchTableBinaryByteLength,
		// });

		return b3dm;
	}

	async loadNode(node){

		if(node.contentLoaded) return;
		if(node.isLoading) return;

		node.isLoading = true;

		let isTileset = node.content.uri.endsWith("json");
		let isBatched = node.content.uri.endsWith("b3dm");
		
		if(isTileset){
			let url = node.tilesetUrl + "/../" + node.content.uri;
			let response = await fetch(url);
			let json = await response.json();

			dbgCount++;

			node.contentLoaded = true;
			this.parseTiles(node, json.root, url);
			// debugger;
			node.isLoading = false;
			
		}else if(isBatched){
			let url = node.tilesetUrl + "/../" + node.content.uri;
			let response = await fetch(url);
			let buffer = await response.arrayBuffer();

			node.content.b3dm = await this.parseB3dm(buffer);
			node.isLoading = false;
			node.contentLoaded = true;
		}

	}

	parseTiles(node, jsNode, tilesetUrl){

		let bv = jsNode.boundingVolume;

		if(bv.sphere){
			let sphere = new BVSphere();
			sphere.position.set(bv.sphere[0], bv.sphere[1], bv.sphere[2]);
			sphere.radius = bv.sphere[3];

			node.boundingVolume = sphere;
		}else{
			console.error("TODO");
		}

		node.content = jsNode.content ?? null;
		node.geometricError = jsNode.geometricError;
		node.contentLoaded = false;
		node.tilesetUrl = tilesetUrl;

		let color = SPECTRAL[(13 * dbgCount) % SPECTRAL.length];
		node.dbgColor = new Vector3(...color);

		let jsChildren = jsNode.children ?? [];
		let i = 0;
		for(let jsChild of jsChildren){

			let childNode = new TDTilesNode();
			node.children.push(childNode);
			childNode.level = node.level;
			childNode.localIndex = i;
			childNode.id = `${node.id}_${i}`;
			childNode.tdtile = this.tiles;

			this.parseTiles(childNode, jsChild, tilesetUrl);
			i++;
		}

	}


	static async load(url){

		let loader = new TDTilesLoader();
		let tiles = new TDTiles(url);

		tiles.loader = loader;
		loader.tiles = tiles;

		let response = await fetch(url);
		let json = await response.json();

		window.dbg_3dtile_json = json;
		loader.parseTiles(tiles.root, json.root, url);

		return tiles;
	}

}