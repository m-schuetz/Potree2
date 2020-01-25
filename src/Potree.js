import { WebGpuRenderer } from "./WebGpuRenderer.js";
import {Scene} from "./Scene.js";
// import {POCLoader} from "./POCLoader.js";

export let renderer = null;
export let scene = new Scene();

// export async function load(url){

// 	if(url.endsWith(".poc")){
// 		return POCLoader.load(url);
// 	}

// 	return null;

// }

export async function setup(canvas){
	console.log("setting up canvas");

	renderer = await WebGpuRenderer.create(canvas);

	console.log("ready to go");
};
















