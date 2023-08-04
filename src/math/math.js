import { Vector3 } from "./Vector3.js";

export * from "./Box3.js";
export * from "./Frustum.js";
export * from "./Line3.js";
export * from "./Matrix4.js";
export * from "./Plane.js";
export * from "./Sphere.js";
export * from "./Ray.js";
export * from "./Vector3.js";
export * from "./Vector4.js";
export * from "./PMath.js";

const _v0 = new Vector3();

// from https://github.com/mrdoob/three.js/blob/dev/src/math/Triangle.js
export function computeNormal(a, b, c){

	let target = new Vector3();

	target.subVectors( c, b );
	_v0.subVectors( a, b );
	target.cross( _v0 );

	const targetLengthSq = target.lengthSq();
	
	if ( targetLengthSq > 0 ) {
		return target.multiplyScalar( 1 / Math.sqrt( targetLengthSq ) );
	}

	return target.set( 0, 0, 0 );

}