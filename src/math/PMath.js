// 
// Adapted from three.js
// license: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
// url: https://github.com/mrdoob/three.js/blob/dev/src/math
//

const π = Math.PI;

export function toRadians(degrees){
	return π * degrees / 180;
}

export function toDegrees(radians){
	return 180 * radians / π;
}

export function ceilN(value, N){
	return value + (N - value % N);
};