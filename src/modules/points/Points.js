
import {SceneNode} from "../../scene/SceneNode.js";
import {Geometry} from "../../core/Geometry.js";

export class Points extends SceneNode{

	constructor(name, geometry){
		super(name);

		this.geometry = geometry;
	}

}