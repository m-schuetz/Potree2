
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
export * from "./scene/SceneNode.js";