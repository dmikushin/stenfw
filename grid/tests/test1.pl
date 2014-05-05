#!/usr/bin/perl -w

my($nx_min) = 1; my($nx_max) = 10;
my($ny_min) = 1; my($ny_max) = 10;
my($ns_min) = 1; my($ns_max) = 10;

my($sx_min) = 1; my($sx_max) = 10;
my($sy_min) = 1; my($sy_max) = 10;
my($ss_min) = 1; my($ss_max) = 10;

my($bx_min) = 1; my($bx_max) = 4;
my($by_min) = 1; my($by_max) = 4;
my($bs_min) = 1; my($bs_max) = 4;

my($ex_min) = 1; my($ex_max) = 4;
my($ey_min) = 1; my($ey_max) = 4;
my($es_min) = 1; my($es_max) = 4;

for ($inx = $nx_min; $inx < $nx_max; $inx++)
{
  for ($iny = $ny_min; $iny < $ny_max; $iny++)
  {
    for ($ins = $ns_min; $ins < $ns_max; $ins++)
    {
      for ($isx = $sx_min; $isx < $sx_max; $isx++)
      {
        for ($isy = $sy_min; $isy < $sy_max; $isy++)
        {
          for ($iss = $ss_min; $iss < $ss_max; $iss++)
          {
            for ($ibx = $bx_min; $ibx < $inx; $ibx++)
            {
              for ($iby = $by_min; $iby < $iny; $iby++)
              {
                for ($ibs = $bs_min; $ibs < $ins; $ibs++)
                {
                  for ($iex = $ex_min; $iex < $inx; $iex++)
                  {
                    for ($iey = $ey_min; $iey < $iny; $iey++)
                    {
                      for ($ies = $es_min; $ies < $ins; $ies++)
                      {
                        my($cmd) = "./grid1 $inx $iny $ins $isx $isy $iss $ibx $iex $iby $iey $ibs $ies 1>> grid1.out 2>> grid1.out";
                        #print "$cmd\n";
                        system($cmd);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }    
    }
  }
}
