MAX_JOBS=10
BASE_DIR="/Users/jmathews/src/github.com/jamathews/pyslidemorpher/assets/audio"

for i in $(seq -w 7 15); do
	(
	    TRACK="Fragment 00${i}"
	    duration=$(ffprobe -v error -show_entries format=duration \
	        -of default=noprint_wrappers=1:nokey=1 \
	        "${BASE_DIR}/${TRACK}/${TRACK}.mp3")

	    echo "${TRACK} = ${duration}s"
	    pipenv run python ./pyslidemorpher.py \
	        "${BASE_DIR}/${TRACK}" \
	        --fps 40 \
	        --seconds-per-transition ${duration} \
	        --hold 0 \
	        --pixel-size 1 \
	        --easing smoothstep \
	        --transition random \
	        --log-level WARNING \
	        --size 3840x2160 \
	        --audio "${BASE_DIR}/${TRACK}/${TRACK}.mp3" \
	        --out "${BASE_DIR}/${TRACK}/${TRACK}.mp4" \
	        2>/dev/null
	) &

	while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
	    sleep 1
	done
done

wait
