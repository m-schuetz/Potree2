
import {PotreeLoader} from "./potree/PotreeLoader.js";
import {PointCloudOctree} from "./potree/PointCloudOctree.js";
import {render} from "./potree/renderQuads.js";


async function load(url){
	let octree = await PotreeLoader.load(url);

	return octree;
}

export let Potree = {
	load: load,
	render: render,
};

export * from "./math/math.js";
export * from "./core/Geometry.js";
export * from "./core/RenderTarget.js";
export * from "./scene/Scene.js";
export * from "./scene/SceneNode.js";
export * from "./scene/Camera.js";
export * from "./navigation/OrbitControls.js";
export * from "./renderer/Renderer.js";
export * from "./modules/points/Points.js";
export * as Timer from "./renderer/Timer.js";