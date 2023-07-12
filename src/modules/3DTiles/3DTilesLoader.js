
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

	async loadNode(node){

		if(node.contentLoaded) return;
		if(node.isLoading) return;

		node.isLoading = true;

		let isTileset = node.content.uri.endsWith("json");
		
		if(isTileset){
			let url = node.tilesetUrl + "/../" + node.content.uri;
			let response = await fetch(url);
			let json = await response.json();

			dbgCount++;
			this.parseTiles(node, json.root, url);

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
		node.tilesetUrl = tilesetUrl;

		let color = SPECTRAL[(13 * dbgCount) % SPECTRAL.length];
		node.dbgColor = new Vector3(...color);

		let jsChildren = jsNode.children ?? [];
		for(let jsChild of jsChildren){

			let childNode = new TDTilesNode();
			node.children.push(childNode);

			this.parseTiles(childNode, jsChild, tilesetUrl);

		}

	}


	static async load(url){

		let loader = new TDTilesLoader();
		let tiles = new TDTiles(url);

		tiles.loader = loader;
		loader.tiles = tiles;

		let response = await fetch(url);
		let json = await response.json();

		loader.parseTiles(tiles.root, json.root, url);

		return tiles;
	}

}