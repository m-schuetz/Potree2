
import {Box3} from "../math/Box3.js";

export class PointCloudOctreeNode{
	constructor(name){
		this.name = name;
		this.loaded = false;
		this.children = new Array(8).fill(null);
		this.level = 0;

		this.boundingBox = new Box3();
	}
}