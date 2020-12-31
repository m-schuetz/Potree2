
const π = Math.PI;

export function toRadians(degrees){
	return π * degrees / 180;
}

export function toDegrees(radians){
	return 180 * radians / π;
}