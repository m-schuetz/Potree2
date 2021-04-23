
import {SceneNode, Geometry} from "potree";

export class Points extends SceneNode{

	constructor(name, geometry){
		super(name);

		this.geometry = geometry;
	}

}