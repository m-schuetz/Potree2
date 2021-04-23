
import {PotreeLoader} from "./potree/PotreeLoader.js";
import {render} from "./potree/renderQuads.js";


async function load(url){
	let octree = await PotreeLoader.load(url);

	return octree;
}


const pickQueue = [];

function pick(x, y, callback){
	pickQueue.push({x, y, callback});
}

export let Potree = {
	load: load,
	render: render,
	pick: pick, pickQueue,
};

export * from "./math/math.js";
export * from "./core/Geometry.js";
export * from "./core/RenderTarget.js";
export * from "./scene/Scene.js";
export * from "./scene/SceneNode.js";
export * from "./modules/mesh/Mesh.js";
export * from "./scene/Camera.js";
export * from "./navigation/OrbitControls.js";
export * from "./renderer/Renderer.js";
export * from "./modules/points/Points.js";
export * as Timer from "./renderer/Timer.js";
export * from "./potree/PointCloudOctree.js";
export * from "./potree/PointCloudOctreeNode.js";
export {render as renderMesh} from "./modules/mesh/renderMesh.js";
export {load as loadGLB} from "./misc/GLBLoader.js";