
import {Vector3, Box3} from "potree";

export class PointCloudOctreeNode{
	constructor(name){
		this.name = name;
		this.loaded = false;
		this.parent = null;
		this.children = new Array(8).fill(null);
		this.level = 0;
		this.numPoints = 0;

		this.boundingBox = new Box3();
	}

	traverse(callback){

		callback(this);

		for(let child of this.children){
			if(child){
				child.traverse(callback);
			}
		}

	}
}