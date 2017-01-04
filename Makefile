test:
	PYTHONPATH='src':'tests' python -m unittest discover -s . -p '*_tests.py'

# Usage: make HOST=127.0.0.1 CUDNN_TARBALL=~/cudnn-8.0-linux-x64-v5.1.tgz ec2_setup
# The cudnn tarball can't be downloaded without being authenticated on NVidia's
# developer portal. It's lame. That's why we need to download locally and then
# upload to remote machine.
# Assumes running Ubuntu on g2.2xlarge or p2.xlarge (or equivalent)
ec2_setup:
	scp $$CUDNN_TARBALL ubuntu@$$HOST:~/
	scp ./src/setup.sh ubuntu@$$HOST:~/
	ssh ubuntu@$$HOST 'sh ~/setup.sh'
	@echo "*** Remote environment is ready. ***"
	@echo "Run 'ssh ubuntu@$$HOST' to start using menc there."
